from datetime import datetime, timezone

import numpy as np

import transit
from transit import angular_separation, refine_transit


def test_empty_start_date_defaults_to_current_utc_datetime(tmp_path, monkeypatch):
    now = datetime(2030, 1, 2, 3, 4, 5, tzinfo=timezone.utc)

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            assert tz is timezone.utc
            return now

    monkeypatch.setattr(transit, "datetime", FixedDateTime)
    config = tmp_path / "config.ini"
    config.write_text(
        "[Observer]\n"
        "latitude = 1\n"
        "longitude = 2\n"
        "[Search]\n"
        "max_distance_km = 25\n"
        "start_date =   \n"
        "end_date = 2030-01-03\n"
        "min_altitude_deg = 10\n"
        "[Targets]\n"
        "transit_objects = Polaris\n"
        "[Satellites]\n"
        "[Output]\n",
        encoding="utf-8",
    )

    assert transit.parse_config(str(config))["start"] == datetime(
        2030, 1, 2, 3, 4, 5
    )


def test_explicit_start_date_is_preserved(tmp_path):
    config = tmp_path / "config.ini"
    config.write_text(
        "[Observer]\n"
        "latitude = 1\n"
        "longitude = 2\n"
        "[Search]\n"
        "max_distance_km = 25\n"
        "start_date = 2029-12-31\n"
        "end_date = 2030-01-03\n"
        "min_altitude_deg = 10\n"
        "[Targets]\n"
        "transit_objects = Polaris\n"
        "[Satellites]\n"
        "[Output]\n",
        encoding="utf-8",
    )

    assert transit.parse_config(str(config))["start"] == datetime(2029, 12, 31)


def test_same_direction_returns_zero():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([1.0, 0.0, 0.0])
    assert angular_separation(a, b) == 0.0


def test_orthogonal_returns_ninety_degrees():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    assert abs(angular_separation(a, b) - 90.0) < 1e-9


def test_bright_star_catalog_includes_checked_j2000_positions():
    assert len(transit.BRIGHT_STARS) == 48
    assert transit.BRIGHT_STARS["Polaris"] == (2.529750, 89.264109)
    assert transit.BRIGHT_STARS["Castor"] == (7.576634, 31.888276)

    polaris = transit.build_target("Polaris", None)
    assert np.isclose(polaris.ra.hours, 2.529750)
    assert np.isclose(polaris.dec.degrees, 89.264109)


def test_sun_uranus_and_neptune_are_supported_targets():
    bodies = {
        "sun": object(),
        "uranus barycenter": object(),
        "neptune barycenter": object(),
    }

    assert transit.build_target("Sun", bodies) is bodies["sun"]
    assert transit.build_target("Uranus", bodies) is bodies["uranus barycenter"]
    assert transit.build_target("Neptune", bodies) is bodies["neptune barycenter"]
    assert transit.TARGET_ANGULAR_RADIUS_RAD["Sun"] > 0.004
    assert transit.TARGET_ANGULAR_RADIUS_RAD["Uranus"] > 0.0
    assert transit.TARGET_ANGULAR_RADIUS_RAD["Neptune"] > 0.0


def test_batched_coordinate_first_vectors_can_be_transposed():
    a = np.array([[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
    b = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    np.testing.assert_allclose(
        angular_separation(a.T, b.T),
        [0.0, 90.0],
        atol=1e-9,
    )


def test_refinement_evaluates_geometry_with_one_element_time_array(monkeypatch):
    ts = transit.load.timescale()
    t_rough = ts.utc(2026, 10, 9, 2, 55, 42.375)
    t_rough_dt = t_rough.utc_datetime()

    def compute_separations(sat, target, earth, observer, times):
        assert times.shape == (1,)
        offset = (times.utc_datetime()[0] - t_rough_dt).total_seconds()
        return np.array([1.0 + offset**2]), None, None, None, np.array([offset**2])

    monkeypatch.setattr(transit, "compute_separations", compute_separations)

    t_ref, sep_km, sep_deg = refine_transit(None, None, None, None, t_rough)

    assert abs((t_ref.utc_datetime() - t_rough_dt).total_seconds()) < 0.001
    assert np.isclose(sep_km, 1.0, atol=1e-6)
    assert sep_deg < 1e-6


def test_corridor_band_preserves_physical_width_at_mid_latitude():
    n = 21
    lat0, lon0 = 41.9028, 12.4964
    lat_axis = np.linspace(lat0 - 0.02, lat0 + 0.02, n)
    lon_axis = np.linspace(lon0 - 0.02, lon0 + 0.02, n)
    lon_grid, lat_grid = np.meshgrid(lon_axis, lat_axis)
    coslat = np.cos(np.radians(lat0))
    sep_grid = (
        10.0
        + (lat_grid - lat0) * 111.32 * 0.6
        + (lon_grid - lon0) * 111.32 * coslat * 0.8
    )
    grid = {
        "grid_shape": (n, n),
        "lats": lat_grid.ravel(),
        "lons": lon_grid.ravel(),
        "sep_km": sep_grid.ravel(),
    }
    corridor_km = 0.05

    center, borders, nearest_point = transit._make_corridor_band(
        grid, corridor_km, lat0, lon0, Z_obs_km=10.0
    )

    assert center[1] == nearest_point
    for point_a, point_b in zip(*borders):
        midpoint_lat = (point_a[0] + point_b[0]) / 2.0
        north_km = (point_a[0] - point_b[0]) * 111.32
        east_km = (
            (point_a[1] - point_b[1])
            * 111.32
            * np.cos(np.radians(midpoint_lat))
        )
        assert np.isclose(np.hypot(north_km, east_km), 2 * corridor_km,
                          atol=1e-5)

    earth_radius_m = 6378137.0

    def to_mercator(point):
        lat, lon = np.radians(point)
        return (
            earth_radius_m * lon,
            earth_radius_m * np.log(np.tan(np.pi / 4 + lat / 2)),
        )

    for point_a, point_b, center_point in zip(*borders, center):
        projected_a = to_mercator(point_a)
        projected_b = to_mercator(point_b)
        projected_center = to_mercator(center_point)
        np.testing.assert_allclose(
            (np.array(projected_a) + projected_b) / 2,
            projected_center,
            atol=1e-7,
        )


def test_nearest_corridor_point_tooltip_includes_event_time(tmp_path, monkeypatch):
    event_time = "2026-10-11 06:00:00.125"
    time = transit.load.timescale().utc(2026, 10, 11, 6, 0, 0.125)
    event = {
        "sat": "ISS",
        "target": "Jupiter",
        "time": time,
        "time_utc": event_time,
        "sep_km": 1.0,
        "sep_deg": 0.01,
        "sat_alt_km": 1000.0,
        "status": "NEAR",
    }
    nearest = (39.25, 9.27)

    monkeypatch.setattr(
        transit,
        "compute_corridor_grid",
        lambda *args, **kwargs: {
            "corridor_km": 0.1,
            "sub_lat": 39.3,
            "sub_lon": 9.3,
        },
    )
    monkeypatch.setattr(
        transit,
        "_make_corridor_band",
        lambda *args, **kwargs: (
            [(39.3, 9.3), nearest, (39.2, 9.2)],
            [[(39.3, 9.3), (39.2, 9.2)], [(39.3, 9.3), (39.2, 9.2)]],
            nearest,
        ),
    )

    path = transit.generate_transit_map(
        event, 41.9028, 12.4964, 25,
        eph_earth=object(), sat=object(), target=object(),
        out_dir=str(tmp_path), tile_layer="osm",
    )

    html = tmp_path.joinpath(path.split("/")[-1]).read_text()
    assert "Jupiter transit (" in html
    assert f" at {event_time} UTC" in html


def test_transit_maps_from_different_sites_do_not_overwrite(tmp_path):
    time = transit.load.timescale().utc(2026, 10, 11, 6)
    event = {
        "sat": "ISS",
        "target": "Capella",
        "time": time,
        "time_utc": "2026-10-11 06:00:00",
        "sep_km": 5.0,
        "sep_deg": 0.01,
        "sat_alt_km": 1000.0,
        "status": "NEAR",
    }

    first = transit.generate_transit_map(
        event, 41.9028, 12.4964, 25, out_dir=str(tmp_path),
        tile_layer="osm",
    )
    second = transit.generate_transit_map(
        event, 41.9037, 12.4964, 25, out_dir=str(tmp_path),
        tile_layer="osm",
    )

    assert first != second
    assert tmp_path.joinpath(first.split("/")[-1]).is_file()
    assert tmp_path.joinpath(second.split("/")[-1]).is_file()
