"""
Space Station Transit Predictor — Verbose Travel Edition

Three-stage search with full diagnostic logging:
    Stage 1: coarse scan (configurable step, default 5 s)
    Stage 2: fine scan around each coarse candidate (1 s)
    Stage 3: golden-section refinement (sub-second)

Horizon filtering happens at THREE levels:
    - coarse scan rejects whole coarse-time samples below horizon
    - fine scan rejects individual fine-time samples below horizon
    - main loop keeps a defensive check before golden-section

Overlapping fine windows from adjacent coarse hits are clustered
before refinement so the same physical event is refined once.

Corridor width uses REAL physics:
    For stars        : half-width = satellite_physical_radius
    For Moon/planets : half-width = (target_ang_rad + sat_ang_rad)
                                   * slant_range * SAFETY

Map geometry:
    The map is ALWAYS centered on the observer.  The corridor is
    drawn using the classifier's sep_km (the exact perpendicular
    distance from the observer to the satellite's LOS at the event
    instant) as the offset from the observer to the corridor axis.
    The axis direction comes from the local gradient of sep_km on
    a tight grid centered on the observer.  This guarantees the
    drawn corridor sits at exactly the same distance the classifier
    reported — no interpolation drift.

    A yellow dot marks the point on the corridor axis nearest the
    observer; a dashed yellow line connects it to the observer marker.

Status semantics:
    CONFIRMED  — satellite crosses YOUR position.
    NEAR       — corridor passes within the search radius but does
                 not cross your exact position.
    REJECT     — target or satellite below the horizon.

Console output uses ANSI colors for the status word:
    CONFIRMED printed in green, NEAR printed in blue, REJECT in grey.

Map output: one self-contained HTML map per event inside the search
radius, named <sat>_<target>_<UTCstamp>_<STATUS>_<tile>_<site>.html

Config file (config.ini):
    [Observer]   latitude, longitude, elevation_m
    [Search]     max_distance_km, start_date, end_date,
                 coarse_step_seconds, fine_step_seconds, min_altitude_deg
    [Targets]    transit_objects
    [Satellites] tle_source
    [Output]     map_dir, tile_layer or tile_layers
"""

import configparser
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import numpy as np
from skyfield.api import load, Star, wgs84
import folium


# ═══════════════════════════════════════════════════════════════════
# ANSI console colors
# ═══════════════════════════════════════════════════════════════════

ANSI_RESET = "\033[0m"
ANSI_GREEN = "\033[32m"
ANSI_BLUE  = "\033[34m"
ANSI_GREY  = "\033[90m"


def colored_status(status):
    if not sys.stdout.isatty():
        return status
    if status == "CONFIRMED":
        return f"{ANSI_GREEN}{status}{ANSI_RESET}"
    if status == "NEAR":
        return f"{ANSI_BLUE}{status}{ANSI_RESET}"
    if status == "REJECT":
        return f"{ANSI_GREY}{status}{ANSI_RESET}"
    return status


def event_condition_tags(event):
    tags = []
    if event.get("daylight"):
        tags.append("DAYLIGHT")
    if event.get("satellite_sunlit") is False:
        tags.append("SATELLITE UNLIT")
    return tags


def format_event_condition_tags(event):
    return "".join(f" [{tag}]" for tag in event_condition_tags(event))


# ═══════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════

# J2000.0 (RA hours, Dec degrees) from HYG Database v4.1; proper-named
# stars with apparent visual magnitude <= 2.0. HYG is CC BY-SA 4.0:
# https://github.com/astronexus/HYG-Database
BRIGHT_STARS = {
    "Sirius":          (6.752481, -16.716116),
    "Canopus":         (6.399195, -52.695660),
    "Arcturus":       (14.261030,  19.182410),
    "Rigil Kentaurus": (14.660765, -60.833976),
    "Vega":           (18.615640,  38.783692),
    "Capella":         (5.278150,  45.997991),
    "Rigel":           (5.242298,  -8.201640),
    "Procyon":         (7.655033,   5.224993),
    "Achernar":        (1.628556, -57.236757),
    "Betelgeuse":      (5.919529,   7.407063),
    "Hadar":          (14.063729, -60.373039),
    "Altair":         (19.846388,   8.868322),
    "Acrux":          (12.443311, -63.099092),
    "Aldebaran":       (4.598677,  16.509301),
    "Spica":          (13.419883, -11.161322),
    "Antares":        (16.490128, -26.432002),
    "Pollux":          (7.755277,  28.026199),
    "Fomalhaut":      (22.960838, -29.622236),
    "Deneb":          (20.690532,  45.280338),
    "Mimosa":         (12.795359, -59.688764),
    "Toliman":        (14.660346, -60.838300),
    "Regulus":        (10.139532,  11.967207),
    "Adhara":          (6.977097, -28.972084),
    "Castor":          (7.576634,  31.888276),
    "Gacrux":         (12.519429, -57.113212),
    "Shaula":         (17.560145, -37.103821),
    "Bellatrix":       (5.418851,   6.349702),
    "Elnath":          (5.438198,  28.607450),
    "Miaplacidus":     (9.220041, -69.717208),
    "Alnilam":         (5.603559,  -1.201920),
    "Alnair":         (22.137209, -46.960975),
    "Alnitak":         (5.679313,  -1.942572),
    "Alioth":         (12.900472,  55.959821),
    "Kaus Australis": (18.402868, -34.384616),
    "Mirfak":          (3.405378,  49.861180),
    "Dubhe":          (11.062155,  61.751033),
    "Wezen":           (7.139857, -26.393200),
    "Alkaid":         (13.792354,  49.313265),
    "Avior":           (8.375236, -59.509483),
    "Sargas":         (17.621980, -42.997824),
    "Menkalinan":      (5.992149,  44.947433),
    "Atria":          (16.811077, -69.027715),
    "Alhena":          (6.628528,  16.399252),
    "Alsephina":       (8.745059, -54.708821),
    "Peacock":        (20.427459, -56.735090),
    "Polaris":         (2.529750,  89.264109),
    "Mirzam":          (6.378329, -17.955918),
    "Alphard":         (9.459790,  -8.658603),
}

TARGET_ANGULAR_RADIUS_RAD = {
    "Moon":    0.00465,
    "Sun":     0.00465047,
    "Venus":   0.0000122,
    "Mars":    0.0000045,
    "Jupiter": 0.0000233,
    "Saturn":  0.0000198,
    "Uranus":  0.0000087,
    "Neptune": 0.0000055,
}

SAT_PHYSICAL_RADIUS_KM = {
    "ISS":    0.050,
    "Tianhe": 0.030,
}
DEFAULT_SAT_PHYSICAL_RADIUS_KM = 0.050

DISK_SAFETY_FACTOR = 5.0

DEFAULT_TLE_URL = (
    "https://celestrak.org/NORAD/elements/gp.php?GROUP=stations&FORMAT=tle"
)

STATUS_COLORS = {
    "CONFIRMED": "green",
    "NEAR":      "blue",
    "REJECT":    "gray",
}

TILE_LAYERS = {
    "esri_imagery": {
        "tiles": ("https://server.arcgisonline.com/ArcGIS/rest/services/"
                  "World_Imagery/MapServer/tile/{z}/{y}/{x}"),
        "attr":  "Tiles © Esri — Source: Esri, Maxar, Earthstar Geographics",
    },
    "esri_street": {
        "tiles": ("https://server.arcgisonline.com/ArcGIS/rest/services/"
                  "World_Street_Map/MapServer/tile/{z}/{y}/{x}"),
        "attr":  "Tiles © Esri",
    },
    "esri_topo": {
        "tiles": ("https://server.arcgisonline.com/ArcGIS/rest/services/"
                  "World_Topo_Map/MapServer/tile/{z}/{y}/{x}"),
        "attr":  "Tiles © Esri",
    },
    "osm": {
        "tiles": "OpenStreetMap",
        "attr":  "© OpenStreetMap contributors",
    },
}

DEFAULT_TILE_LAYER = "esri_imagery"


# ═══════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════

def _parse_tile_layers(cfg):
    raw = ""
    if cfg.has_option("Output", "tile_layers"):
        raw = cfg.get("Output", "tile_layers")
    elif cfg.has_option("Output", "tile_layer"):
        raw = cfg.get("Output", "tile_layer")

    names = [n.strip() for n in raw.split(",") if n.strip()]
    valid = [n for n in names if n in TILE_LAYERS]
    unknown = [n for n in names if n not in TILE_LAYERS]

    if unknown:
        print(f"  Warning: unknown tile layer(s) ignored: {unknown}")
        print(f"  Known layers: {list(TILE_LAYERS.keys())}")

    if not valid:
        valid = [DEFAULT_TILE_LAYER]
    return valid


def parse_config(path="config.ini"):
    cfg = configparser.ConfigParser()
    try:
        if not cfg.read(path):
            raise FileNotFoundError(f"Config file not found: {path}")
    except configparser.Error as e:
        raise SystemExit(
            f"Error parsing {path}: {e}\n"
            f"The file must start with a section header like [Observer]."
        )

    start_date = cfg.get("Search", "start_date", fallback="").strip()
    start = (
        datetime.fromisoformat(start_date)
        if start_date
        else datetime.now(timezone.utc).replace(tzinfo=None)
    )

    return {
        "lat":           cfg.getfloat("Observer", "latitude"),
        "lon":           cfg.getfloat("Observer", "longitude"),
        "elev":          cfg.getfloat("Observer", "elevation_m",
                                      fallback=0.0),
        "max_dist_km":   cfg.getfloat("Search", "max_distance_km"),
        "start":         start,
        "end":           datetime.fromisoformat(
                             cfg.get("Search", "end_date")),
        "coarse_step_s": cfg.getfloat("Search", "coarse_step_seconds",
                                      fallback=5.0),
        "fine_step_s":   cfg.getfloat("Search", "fine_step_seconds",
                                      fallback=1.0),
        "min_alt":       cfg.getfloat("Search", "min_altitude_deg"),
        "targets":       [t.strip() for t in
                          cfg.get("Targets", "transit_objects").split(",")
                          if t.strip()],
        "tle_source":    cfg.get("Satellites", "tle_source",
                                 fallback=DEFAULT_TLE_URL),
        "map_dir":       cfg.get("Output", "map_dir",
                                 fallback="transit_maps"),
        "tile_layers":   _parse_tile_layers(cfg),
    }


# ═══════════════════════════════════════════════════════════════════
# Satellites
# ═══════════════════════════════════════════════════════════════════

def load_satellites(source_url):
    print(f"  Fetching TLEs from {source_url}")
    t0 = time.time()
    tle_lines = load.tle_file(source_url)
    print(f"  Parsed {len(tle_lines)} satellites in {time.time()-t0:.1f}s")

    sats = {}
    for sat in tle_lines:
        n = sat.name.upper()
        if "ISS" in n or "ZARYA" in n:
            sats.setdefault("ISS", sat)
        elif "TIANHE" in n or "CSS" in n:
            sats.setdefault("Tianhe", sat)

    if not sats:
        raise RuntimeError("No ISS or Tianhe satellites found in TLE data.")

    print("  TLE epochs:")
    for name, sat in sats.items():
        try:
            epoch = sat.epoch.utc_datetime()
            now = datetime.now()
            if getattr(epoch, "tzinfo", None) is not None:
                now = datetime.now(epoch.tzinfo)
            age_days = (now - epoch).total_seconds() / 86400.0
            print(f"    {name:8s} epoch {epoch.strftime('%Y-%m-%d %H:%M')} UTC"
                  f"  (age {age_days:+.1f} days)")
        except Exception as e:
            print(f"    {name:8s} epoch unreadable: {e}")

    return sats


# ═══════════════════════════════════════════════════════════════════
# Targets
# ═══════════════════════════════════════════════════════════════════

def build_target(name, eph):
    if name == "Moon":
        return eph["moon"]
    if name in ("Sun", "Venus", "Mars", "Jupiter", "Saturn",
                "Uranus", "Neptune"):
        key = {
            "Sun":     "sun",
            "Venus":   "venus",
            "Mars":    "mars barycenter",
            "Jupiter": "jupiter barycenter",
            "Saturn":  "saturn barycenter",
            "Uranus":  "uranus barycenter",
            "Neptune": "neptune barycenter",
        }[name]
        return eph[key]
    if name in BRIGHT_STARS:
        ra_h, dec_d = BRIGHT_STARS[name]
        return Star(ra_hours=ra_h, dec_degrees=dec_d)
    raise ValueError(f"Unknown target: {name}")


def is_star(name):
    return name in BRIGHT_STARS


def disk_corridor_km(target_name, slant_range_km, sat_name=None):
    sat_rad_km = SAT_PHYSICAL_RADIUS_KM.get(
        sat_name, DEFAULT_SAT_PHYSICAL_RADIUS_KM
    )

    if is_star(target_name):
        return sat_rad_km

    if slant_range_km <= 0:
        return 0.0
    target_ang_rad = TARGET_ANGULAR_RADIUS_RAD.get(target_name, 0.0)
    sat_ang_rad = sat_rad_km / slant_range_km
    return (target_ang_rad + sat_ang_rad) * slant_range_km * DISK_SAFETY_FACTOR


# ═══════════════════════════════════════════════════════════════════
# Line of sight
# ═══════════════════════════════════════════════════════════════════

def los_unit_vectors(target, earth, observer, times):
    observer_ssb = earth + observer
    apparent = observer_ssb.at(times).observe(target).apparent()
    vec = np.array(apparent.position.km, dtype=float, copy=True)
    norm = np.linalg.norm(vec, axis=0)
    norm = np.where(norm == 0.0, 1.0, norm)
    unit = vec / norm
    if isinstance(target, Star):
        rng = np.full(vec.shape[1], np.inf)
    else:
        rng = norm
    return unit, rng


def angular_separation(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.ndim > 1 and a.shape[0] != 3 and a.shape[-1] == 3:
        vector_axis = -1
    else:
        vector_axis = 0

    a_norm = np.linalg.norm(a, axis=vector_axis, keepdims=True)
    b_norm = np.linalg.norm(b, axis=vector_axis, keepdims=True)
    a_norm = np.where(a_norm == 0.0, 1.0, a_norm)
    b_norm = np.where(b_norm == 0.0, 1.0, b_norm)
    cos = np.clip(
        np.sum((a / a_norm) * (b / b_norm), axis=vector_axis),
        -1.0,
        1.0,
    )
    return np.degrees(np.arccos(cos))


def compute_separations(sat, target, earth, observer, times):
    los_unit, target_range = los_unit_vectors(target, earth, observer, times)
    sat_pos = np.array(
        (sat - observer).at(times).position.km, dtype=float, copy=True
    )
    slant_range = np.linalg.norm(sat_pos, axis=0)
    proj = np.sum(sat_pos * los_unit, axis=0)
    perp = sat_pos - los_unit * proj
    sep_km = np.linalg.norm(perp, axis=0)

    sat_dir = sat_pos / np.where(slant_range == 0.0, 1.0, slant_range)
    sep_deg = angular_separation(sat_dir, los_unit)
    return sep_km, slant_range, proj, target_range, sep_deg


def horizon_mask(sat, target, earth, observer, times, min_alt_deg):
    return (target_horizon_mask(target, earth, observer, times, min_alt_deg)
            & satellite_horizon_mask(sat, observer, times))


def target_horizon_mask(target, earth, observer, times, min_alt_deg):
    observer_ssb = earth + observer
    tgt_alt, _, _ = (observer_ssb.at(times)
                     .observe(target).apparent().altaz())
    return tgt_alt.degrees >= min_alt_deg


def satellite_horizon_mask(sat, observer, times):
    sat_alt_ao, _, _ = (sat - observer).at(times).altaz()
    return sat_alt_ao.degrees >= 0.0


# ═══════════════════════════════════════════════════════════════════
# Horizon pre-filter
# ═══════════════════════════════════════════════════════════════════

def target_ever_visible(target, earth, observer, t_start, t_end,
                        min_alt, sample_hours=0.25):
    ts = load.timescale()
    total_s = (t_end - t_start).total_seconds()
    n = max(2, int(total_s / (sample_hours * 3600.0)) + 1)
    offsets = np.linspace(0.0, total_s, n)
    times = ts.utc(
        t_start.year, t_start.month, t_start.day,
        t_start.hour, t_start.minute, t_start.second + offsets,
    )
    observer_ssb = earth + observer
    alt, az, _ = observer_ssb.at(times).observe(target).apparent().altaz()
    return bool(np.any(alt.degrees >= min_alt)), float(np.max(alt.degrees))


# ═══════════════════════════════════════════════════════════════════
# Shared coarse grid + caches
# ═══════════════════════════════════════════════════════════════════

def build_coarse_grid(t_start, t_end, step_s):
    ts = load.timescale()
    total_s = (t_end - t_start).total_seconds()
    n_steps = int(total_s / step_s)
    if n_steps < 2:
        return None, 0
    secs = np.arange(n_steps) * step_s
    times = ts.utc(
        t_start.year, t_start.month, t_start.day,
        t_start.hour, t_start.minute, t_start.second + secs,
    )
    return times, n_steps


def cache_target_los(targets, earth, observer, times, min_alt_deg=0.0):
    print("  Caching target line-of-sight over coarse grid …")
    cache = {}
    for name, target in targets.items():
        t0 = time.time()
        observer_ssb = earth + observer
        apparent = observer_ssb.at(times).observe(target).apparent()
        vec = np.array(apparent.position.km, dtype=float, copy=True)
        norm = np.linalg.norm(vec, axis=0)
        safe_norm = np.where(norm == 0.0, 1.0, norm)
        unit = vec / safe_norm
        rng = np.full(norm.shape, np.inf) if isinstance(target, Star) else norm
        altitude, _, _ = apparent.altaz()
        cache[name] = {
            "unit": unit,
            "range": rng,
            "above_horizon": altitude.degrees >= min_alt_deg,
        }
        print(f"    {name:12s} LOS cached in {time.time()-t0:.2f}s")
    return cache


def cache_satellite_positions(sats, observer, times):
    print("  Caching satellite positions over coarse grid …")
    cache = {}
    for name, sat in sats.items():
        t0 = time.time()
        sat_state = (sat - observer).at(times)
        pos = np.array(sat_state.position.km, dtype=float, copy=True)
        alt = np.linalg.norm(pos, axis=0)
        altitude, _, _ = sat_state.altaz()
        cache[name] = {
            "pos": pos,
            "alt": alt,
            "above_horizon": altitude.degrees >= 0.0,
        }
        print(f"    {name:12s} positions cached in {time.time()-t0:.2f}s")
    return cache


def coarse_pair_scan(sat_entry, target_entry, times, max_dist_km,
                     earth=None, observer=None, sat=None, target=None,
                     min_alt_deg=0.0, above_horizon=None,
                     label="", verbose=True):
    sat_pos = sat_entry["pos"]
    sat_alt = sat_entry["alt"]
    tgt_unit = target_entry["unit"]
    tgt_range = target_entry["range"]

    proj = np.sum(sat_pos * tgt_unit, axis=0)
    perp = sat_pos - tgt_unit * proj
    sep_km = np.linalg.norm(perp, axis=0)
    sat_dir = sat_pos / np.where(sat_alt == 0.0, 1.0, sat_alt)
    sep_deg = angular_separation(sat_dir, tgt_unit)

    coarse_net_deg = np.degrees(np.minimum(max_dist_km / sat_alt, 1.0))
    mask = (sep_deg < coarse_net_deg) & (proj > 0)
    finite_rng = np.isfinite(tgt_range)
    mask &= (~finite_rng) | (proj < tgt_range)

    if above_horizon is not None:
        mask &= above_horizon
    elif earth is not None and observer is not None \
            and sat is not None and target is not None:
        mask &= horizon_mask(sat, target, earth, observer,
                             times, min_alt_deg)

    idx = np.where(mask)[0]

    if verbose and len(idx) > 0:
        print(f"      [{label}] coarse hits (within {max_dist_km:.1f} km):",
              flush=True)
        for i in idx:
            print(f"        {times[i].utc_strftime('%Y-%m-%d %H:%M:%S')}  "
                  f"sep={sep_km[i]:8.3f} km  sep_angle={sep_deg[i]:8.4f}°  "
                  f"slant={sat_alt[i]:.0f} km",
                  flush=True)

    candidates = []
    for i in idx:
        candidates.append({
            "time": times[i],
            "separation_km": float(sep_km[i]),
            "separation_deg": float(sep_deg[i]),
            "sat_alt_km": float(sat_alt[i]),
        })
    return candidates, len(idx)


# ═══════════════════════════════════════════════════════════════════
# Stage 2 — fine scan
# ═══════════════════════════════════════════════════════════════════

def scan_fine(sat, target, earth, observer, coarse_candidates,
              fine_step_s, half_window_s, max_dist_km, label="",
              verbose=True, min_alt_deg=0.0):
    ts = load.timescale()
    fine_candidates = []

    for ci, c in enumerate(coarse_candidates):
        t_center = c["time"].utc_datetime()
        offsets = np.arange(-half_window_s, half_window_s + fine_step_s,
                            fine_step_s)
        times = ts.utc(
            t_center.year, t_center.month, t_center.day,
            t_center.hour, t_center.minute,
            t_center.second + offsets,
        )

        sep_km, slant, proj, target_range, sep_deg = compute_separations(
            sat, target, earth, observer, times
        )

        above = horizon_mask(sat, target, earth, observer,
                             times, min_alt_deg)

        max_angle_deg = np.degrees(np.minimum(max_dist_km / slant, 1.0))
        mask = (sep_deg < max_angle_deg) & (proj > 0) & above
        finite_rng = np.isfinite(target_range)
        mask &= (~finite_rng) | (proj < target_range)

        if verbose:
            n_above = int(np.sum(above))
            n_in_window = int(np.sum(mask))
            min_sep = float(np.min(sep_km))
            min_idx = int(np.argmin(sep_km))
            print(f"        fine window #{ci+1} around "
                  f"{t_center.strftime('%Y-%m-%d %H:%M:%S')}: "
                  f"min sep = {min_sep:.4f} km at "
                  f"{times[min_idx].utc_strftime('%H:%M:%S')}  "
                  f"({n_above}/{len(times)} above horizon, "
                  f"{n_in_window} candidates)",
                  flush=True)

        for i in np.where(mask)[0]:
            fine_candidates.append({
                "time": times[i],
                "separation_km": float(sep_km[i]),
                "separation_deg": float(sep_deg[i]),
                "sat_alt_km": float(slant[i]),
            })

    print(f"      [{label}] fine {fine_step_s:.0f} s step over "
          f"±{half_window_s:.0f} s × {len(coarse_candidates)} windows "
          f"→ {len(fine_candidates)} candidates", flush=True)
    return fine_candidates


def cluster_candidates(candidates, cluster_gap_s):
    if not candidates:
        return []
    s = sorted(candidates, key=lambda c: c["time"].utc_datetime())
    out = [s[0]]
    for c in s[1:]:
        dt = (c["time"].utc_datetime()
              - out[-1]["time"].utc_datetime()).total_seconds()
        if dt <= cluster_gap_s:
            if c["separation_km"] < out[-1]["separation_km"]:
                out[-1] = c
        else:
            out.append(c)
    return out


# ═══════════════════════════════════════════════════════════════════
# Stage 3 — golden-section refinement
# ═══════════════════════════════════════════════════════════════════

def refine_transit(sat, target, earth, observer, t_rough, window_s=2.0):
    ts = load.timescale()

    coarse = np.linspace(-window_s, window_s, 21)
    rough_dt = t_rough.utc_datetime()
    coarse_times = ts.from_datetimes([
        rough_dt + timedelta(seconds=float(offset))
        for offset in coarse
    ])
    coarse_sep_deg = compute_separations(
        sat, target, earth, observer, coarse_times
    )[4]
    best = int(np.argmin(coarse_sep_deg))
    lo = coarse[max(0, best - 1)]
    hi = coarse[min(len(coarse) - 1, best + 1)]

    phi = (np.sqrt(5.0) - 1.0) / 2.0
    for _ in range(20):
        c = hi - phi * (hi - lo)
        d = lo + phi * (hi - lo)
        pair_times = ts.from_datetimes([
            rough_dt + timedelta(seconds=float(c)),
            rough_dt + timedelta(seconds=float(d)),
        ])
        pair_sep_deg = compute_separations(
            sat, target, earth, observer, pair_times
        )[4]
        if pair_sep_deg[0] < pair_sep_deg[1]:
            hi = d
        else:
            lo = c
    t_opt_s = 0.5 * (lo + hi)

    t_opt_dt = rough_dt + timedelta(seconds=float(t_opt_s))
    t_opt = ts.from_datetime(t_opt_dt)
    opt_times = ts.from_datetimes([t_opt_dt])
    sep_km, _, _, _, sep_deg = compute_separations(
        sat, target, earth, observer, opt_times
    )
    sep_km_opt, sep_deg_opt = float(sep_km[0]), float(sep_deg[0])
    return t_opt, sep_km_opt, sep_deg_opt


# ═══════════════════════════════════════════════════════════════════
# Deduplicate
# ═══════════════════════════════════════════════════════════════════

def deduplicate(events, min_gap_s=30.0):
    if not events:
        return events
    events = sorted(events, key=lambda x: x["time"].utc_datetime())
    kept = [events[0]]
    for e in events[1:]:
        dt = (e["time"].utc_datetime()
              - kept[-1]["time"].utc_datetime()).total_seconds()
        if dt < min_gap_s:
            cur = kept[-1]
            better = (
                (e["status"] == "CONFIRMED" and cur["status"] != "CONFIRMED")
                or (e["status"] == cur["status"] and e["sep_km"] < cur["sep_km"])
            )
            if better:
                kept[-1] = e
        else:
            kept.append(e)
    return kept


# ═══════════════════════════════════════════════════════════════════
# Corridor geometry
# ═══════════════════════════════════════════════════════════════════

def _observer_grid_around(lat0, lon0, radius_km, n):
    dlat = radius_km / 111.32
    dlon = radius_km / (111.32 * max(np.cos(np.radians(lat0)), 1e-6))
    lats = np.linspace(lat0 - dlat, lat0 + dlat, n)
    lons = np.linspace(lon0 - dlon, lon0 + dlon, n)
    LON, LAT = np.meshgrid(lons, lats)
    return LAT.ravel(), LON.ravel()


def compute_corridor_grid(sat, target, earth, t_event,
                          lat0, lon0, radius_km,
                          target_name="", sat_name=None, n=201,
                          sample_radius_km=None):
    """
    Compute sep_km on a grid of observer positions around (lat0, lon0)
    at the event instant.  Fractional seconds are preserved so the
    grid is at the exact same instant as the classifier.
    """
    ts = load.timescale()

    sample_radius = (radius_km if sample_radius_km is None
                     else min(radius_km, sample_radius_km))
    lats, lons = _observer_grid_around(lat0, lon0, sample_radius, n=n)
    N = lats.size
    t_event_dt = t_event.utc_datetime()

    sec_frac = t_event_dt.second + t_event_dt.microsecond / 1e6

    times = ts.utc(
        t_event_dt.year, t_event_dt.month, t_event_dt.day,
        t_event_dt.hour, t_event_dt.minute,
        sec_frac + np.zeros(N),
    )

    obs = wgs84.latlon(lats, lons)
    observer_ssb = earth + obs
    obs_pos_ssb = np.array(
        observer_ssb.at(times).position.km, dtype=float, copy=True
    )
    sat_pos_ssb = np.array(
        (earth + sat).at(times).position.km, dtype=float, copy=True
    )
    sat_vec = sat_pos_ssb - obs_pos_ssb

    tgt_vec = np.array(
        observer_ssb.at(times).observe(target).apparent().position.km,
        dtype=float, copy=True,
    )
    tgt_norm = np.linalg.norm(tgt_vec, axis=0)
    tgt_norm = np.where(tgt_norm == 0.0, 1.0, tgt_norm)
    tgt_unit = tgt_vec / tgt_norm

    proj = np.sum(sat_vec * tgt_unit, axis=0)
    perp = sat_vec - tgt_unit * proj
    sep_km = np.linalg.norm(perp, axis=0)

    ref_obs = wgs84.latlon(lat0, lon0)
    ref_pos_ssb = np.array(
        (earth + ref_obs).at(times).position.km, dtype=float, copy=True
    )
    slant_ref = float(np.linalg.norm(
        sat_pos_ssb[:, 0] - ref_pos_ssb[:, 0]
    ))

    corridor_km = disk_corridor_km(target_name, slant_ref,
                                   sat_name=sat_name)

    t_scalar = ts.utc(
        t_event_dt.year, t_event_dt.month, t_event_dt.day,
        t_event_dt.hour, t_event_dt.minute,
        sec_frac,
    )
    sub = wgs84.subpoint_of(sat.at(t_scalar))
    sub_lat = float(np.atleast_1d(sub.latitude.degrees)[0])
    sub_lon = float(np.atleast_1d(sub.longitude.degrees)[0])

    return {
        "lats": lats,
        "lons": lons,
        "sep_km": sep_km,
        "corridor_km": corridor_km,
        "grid_shape": (n, n),
        "extent_radius_km": radius_km,
        "sub_lat": sub_lat,
        "sub_lon": sub_lon,
        "slant_range_km": slant_ref,
    }


def _make_corridor_band(grid, corridor_km, obs_lat, obs_lon,
                        Z_obs_km=None):
    """
    Build the corridor band anchored at the OBSERVER's nearest point
    on the corridor axis.

    If Z_obs_km is provided, it is used directly as the perpendicular
    distance from the observer to the corridor axis.  Otherwise, this
    distance is bilinearly interpolated from the grid (used only as
    a fallback; the caller normally supplies the classifier's sep_km
    so the map agrees exactly with the printed event log).

    The direction of the corridor axis is always taken from the
    local gradient of sep_km on the grid, evaluated at the grid cell
    containing the observer.  That gives the orientation of the
    corridor at the observer's position.
    """
    n = grid["grid_shape"][0]
    Z = grid["sep_km"].reshape(n, n)
    LAT = grid["lats"].reshape(n, n)
    LON = grid["lons"].reshape(n, n)

    lat_axis = LAT[:, 0]
    lon_axis = LON[0, :]

    j = int(np.clip(np.searchsorted(lat_axis, obs_lat) - 1, 0, n - 2))
    i = int(np.clip(np.searchsorted(lon_axis, obs_lon) - 1, 0, n - 2))

    lat_a = lat_axis[j]
    lat_b = lat_axis[j + 1]
    lon_a = lon_axis[i]
    lon_b = lon_axis[i + 1]

    fy = float(np.clip((obs_lat - lat_a) / max(lat_b - lat_a, 1e-12),
                       0.0, 1.0))
    fx = float(np.clip((obs_lon - lon_a) / max(lon_b - lon_a, 1e-12),
                       0.0, 1.0))

    Z11 = Z[j,     i]
    Z12 = Z[j,     i + 1]
    Z21 = Z[j + 1, i]
    Z22 = Z[j + 1, i + 1]

    if Z_obs_km is None:
        Z_obs = ((1 - fy) * (1 - fx) * Z11
                 + (1 - fy) * fx * Z12
                 + fy * (1 - fx) * Z21
                 + fy * fx * Z22)
    else:
        Z_obs = float(Z_obs_km)

    dZ_dlon = ((1 - fy) * (Z12 - Z11) + fy * (Z22 - Z21)) / \
              max(lon_b - lon_a, 1e-12)
    dZ_dlat = ((1 - fx) * (Z21 - Z11) + fx * (Z22 - Z12)) / \
              max(lat_b - lat_a, 1e-12)

    coslat = max(np.cos(np.radians(obs_lat)), 1e-6)
    g_east  = dZ_dlon / (111.32 * coslat)
    g_north = dZ_dlat / 111.32

    gmag = float(np.hypot(g_north, g_east))
    if gmag < 1e-12:
        u_north, u_east = 0.0, 1.0
    else:
        u_north = -g_east / gmag
        u_east  =  g_north / gmag

    u_lat = u_north / 111.32
    u_lon = u_east / (111.32 * coslat)

    g_hat_north = g_north / max(gmag, 1e-12)
    g_hat_east  = g_east  / max(gmag, 1e-12)
    disp_north = -Z_obs * g_hat_north
    disp_east  = -Z_obs * g_hat_east
    disp_lat = disp_north / 111.32
    disp_lon = disp_east  / (111.32 * coslat)

    axis_lat = obs_lat + disp_lat
    axis_lon = obs_lon + disp_lon

    # Extend the drawn lines well beyond the tight grid so they
    # cross the entire map frame at any angle.
    lat0_deg = float(LAT[0, 0])
    lat1_deg = float(LAT[-1, 0])
    lon0_deg = float(LON[0, 0])
    lon1_deg = float(LON[0, -1])
    extent_radius_km = grid.get("extent_radius_km")
    if extent_radius_km is None:
        half_lat = abs(lat1_deg - lat0_deg) / 2.0
        half_lon = abs(lon1_deg - lon0_deg) / 2.0
    else:
        half_lat = extent_radius_km / 111.32
        half_lon = extent_radius_km / (
            111.32 * max(np.cos(np.radians(obs_lat)), 1e-6)
        )
    u_mag = float(np.hypot(u_lat, u_lon))
    if u_mag < 1e-12:
        t_extent = 3.0 * float(np.hypot(half_lat, half_lon))
    else:
        t_extent = 3.0 * float(np.hypot(half_lat, half_lon)) / u_mag

    p1 = (axis_lat - u_lat * t_extent, axis_lon - u_lon * t_extent)
    p2 = (axis_lat + u_lat * t_extent, axis_lon + u_lon * t_extent)
    nearest_point = (axis_lat, axis_lon)
    center = [p1, nearest_point, p2]

    earth_radius_m = 6378137.0

    def to_mercator(point):
        point_lat = float(np.clip(point[0], -85.0, 85.0))
        x = earth_radius_m * np.radians(point[1])
        y = earth_radius_m * np.log(np.tan(
            np.pi / 4.0 + np.radians(point_lat) / 2.0
        ))
        return x, y

    def from_mercator(x, y):
        point_lon = np.degrees(x / earth_radius_m)
        point_lat = np.degrees(
            2.0 * np.arctan(np.exp(y / earth_radius_m)) - np.pi / 2.0
        )
        return point_lat, point_lon

    projected_center = [to_mercator(point) for point in center]
    border_a = []
    border_b = []
    for i, point in enumerate(center):
        prev_x, prev_y = projected_center[max(i - 1, 0)]
        next_x, next_y = projected_center[min(i + 1, len(center) - 1)]
        tangent_x = next_x - prev_x
        tangent_y = next_y - prev_y
        tangent_mag = float(np.hypot(tangent_x, tangent_y))
        if tangent_mag < 1e-12:
            normal_x, normal_y = 1.0, 0.0
        else:
            normal_x = -tangent_y / tangent_mag
            normal_y = tangent_x / tangent_mag

        point_coslat = max(np.cos(np.radians(point[0])), 1e-6)
        offset_m = corridor_km * 1000.0 / point_coslat
        point_x, point_y = projected_center[i]
        border_a.append(from_mercator(
            point_x + normal_x * offset_m,
            point_y + normal_y * offset_m,
        ))
        border_b.append(from_mercator(
            point_x - normal_x * offset_m,
            point_y - normal_y * offset_m,
        ))

    return center, [border_a, border_b], nearest_point


# ═══════════════════════════════════════════════════════════════════
# Map helpers
# ═══════════════════════════════════════════════════════════════════

def _safe_name(text):
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in text)


def _fmt_km(km):
    if km is None:
        return "?"
    if km < 0.001:
        return f"{km * 1e6:.0f} mm"
    if km < 1.0:
        return f"{km * 1000:.0f} m"
    return f"{km:.2f} km"


def _add_legend(m, confirmed_count, near_count, tile_layer_name,
                has_corridor=False, physical_km=None, event_tags=None):
    corridor_row = ""
    if has_corridor:
        phys = _fmt_km(physical_km)
        corridor_row = f"""
      <div style="margin-top: 6px; padding-top: 6px;
                  border-top: 1px solid #ccc;">
        <div style="display: flex; align-items: center; margin-bottom: 3px;">
          <span style="display:inline-block; width:16px; height:3px;
                       background:orange; margin-right:8px;"></span>
          Corridor borders (±{phys}, true width)
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 3px;">
          <span style="display:inline-block; width:16px; height:3px;
                       background:gold; margin-right:8px;"></span>
          Center line (corridor axis)
        </div>
        <div style="display: flex; align-items: center;">
          <span style="display:inline-block; width:12px; height:12px;
                       border-radius:50%; background:#ffea00;
                       border:2px solid #222; margin-right:8px;"></span>
          Nearest point of corridor to you
        </div>
      </div>
    """
    legend_html = f"""
    <div style="position: fixed; bottom: 24px; left: 24px; z-index: 9999;
                background: rgba(255,255,255,0.92);
                padding: 10px 14px; border-radius: 6px;
                font-family: sans-serif; font-size: 13px;
                box-shadow: 0 1px 4px rgba(0,0,0,0.4); max-width: 380px;">
      <div style="font-weight: bold; margin-bottom: 6px;">
        Transit status
      </div>
      {event_tags or ""}
      <div style="display: flex; align-items: center; margin-bottom: 3px;">
        <span style="display:inline-block; width:12px; height:12px;
                     border-radius:50%; background:{STATUS_COLORS['CONFIRMED']};
                     margin-right:8px; border:2px solid #222;"></span>
        CONFIRMED — satellite crosses your position ({confirmed_count})
      </div>
      <div style="display: flex; align-items: center;">
        <span style="display:inline-block; width:12px; height:12px;
                     border-radius:50%; background:{STATUS_COLORS['NEAR']};
                     margin-right:8px; border:2px solid #222;"></span>
        NEAR — corridor within search radius, misses you ({near_count})
      </div>
      {corridor_row}
      <div style="margin-top: 6px; color: #555; font-size: 11px;">
        tiles: {tile_layer_name}
      </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))


def _zoom_for_radius(radius_km):
    if radius_km <= 0.3:
        return 15
    if radius_km <= 0.6:
        return 15
    if radius_km <= 1.0:
        return 14
    if radius_km <= 2.0:
        return 14
    if radius_km <= 5.0:
        return 13
    if radius_km <= 15.0:
        return 12
    if radius_km <= 40.0:
        return 11
    if radius_km <= 100.0:
        return 10
    if radius_km <= 300.0:
        return 9
    return 8


# ═══════════════════════════════════════════════════════════════════
# Map generation
# ═══════════════════════════════════════════════════════════════════

def generate_transit_map(event, lat, lon, max_dist_km,
                         eph_earth=None, sat=None, target=None,
                         out_dir="transit_maps",
                         tile_layer="esri_imagery"):
    os.makedirs(out_dir, exist_ok=True)

    layer = TILE_LAYERS.get(tile_layer, TILE_LAYERS[DEFAULT_TILE_LAYER])
    status = event.get("status", "NEAR")
    color = STATUS_COLORS.get(status, "gray")
    condition_tags = event_condition_tags(event)
    condition_badges = "".join(
        f'<span style="display:inline-block; margin:0 5px 5px 0; '
        f'padding:3px 7px; border-radius:4px; background:#fff3cd; '
        f'color:#664d03; font-weight:bold;">{tag}</span>'
        for tag in condition_tags
    )
    condition_popup = "".join(
        f"<br><b>{tag}</b>" for tag in condition_tags
    )

    stamp = event["time"].utc_strftime("%Y%m%dT%H%M%SZ")
    site = (f"lat{lat:+.6f}_lon{lon:+.6f}"
            .replace("+", "p").replace("-", "m").replace(".", "p"))
    fname = (f"{_safe_name(event['sat'])}_"
             f"{_safe_name(event['target'])}_"
             f"{stamp}_{status}_{_safe_name(tile_layer)}_{site}.html")
    out_html = os.path.join(out_dir, fname)

    corridor_km = event.get("corridor_km")
    has_corridor = False

    # The classifier's separation from the observer's LOS to the
    # satellite.  This is the authoritative value that will be used
    # as the corridor offset.
    obs_sep_km = float(event.get("sep_km", 0.0))

    if corridor_km is None:
        corridor_km = disk_corridor_km(
            event["target"], event["sat_alt_km"], sat_name=event["sat"]
        )

    # ---------- Choose view size so both observer and corridor fit ----------
    if corridor_km > 0:
        view_km = max(obs_sep_km, corridor_km * 20.0)
        tight_radius_km = min(max(view_km * 1.5, 0.5), max_dist_km)
    else:
        tight_radius_km = min(max(obs_sep_km * 1.5, 2.0), max_dist_km)

    sample_radius_km = min(
        tight_radius_km, max(corridor_km, 0.05)
    )

    # Map is ALWAYS centered on the observer
    m = folium.Map(
        location=[lat, lon],
        zoom_start=_zoom_for_radius(tight_radius_km),
        tiles=layer["tiles"],
        attr=layer["attr"],
        control_scale=True,
    )

    # Sample locally for the corridor direction, but extend the drawn lines
    # across the full map view.
    if eph_earth is not None and sat is not None and target is not None \
            and corridor_km > 0:
        try:
            grid = compute_corridor_grid(
                sat, target, eph_earth, event["time"],
                lat, lon, tight_radius_km,
                target_name=event["target"],
                sat_name=event["sat"], n=5,
                sample_radius_km=sample_radius_km,
            )
            corridor_km = grid["corridor_km"]

            print(f"      tight grid: r={tight_radius_km:.3f} km "
                  f"n=5 local  physical ±{_fmt_km(corridor_km)}  "
                  f"obs_sep={_fmt_km(obs_sep_km)}",
                  flush=True)

            # Pass the classifier's sep_km so the map agrees exactly
            # with the printed event log.
            center_line, border_lines, nearest_point = \
                _make_corridor_band(grid, corridor_km, lat, lon,
                                    Z_obs_km=obs_sep_km)

            for line in border_lines:
                folium.PolyLine(
                    line, color="#000000", weight=8, opacity=0.7,
                    tooltip=(f"Corridor border "
                             f"(±{_fmt_km(corridor_km)}, true width)"),
                ).add_to(m)
                folium.PolyLine(
                    line, color="#ff6600", weight=4, opacity=1.0,
                ).add_to(m)

            for line in [center_line]:
                folium.PolyLine(
                    line, color="#000000", weight=8, opacity=0.7,
                    tooltip="Center line (corridor axis)",
                ).add_to(m)
                folium.PolyLine(
                    line, color="#ffd700", weight=4, opacity=1.0,
                ).add_to(m)

            n_lat, n_lon = nearest_point
            coslat_n = max(np.cos(np.radians(n_lat)), 1e-6)
            dkm = float(np.hypot((n_lat - lat) * 111.32,
                                 (n_lon - lon) * 111.32 * coslat_n))
            folium.PolyLine(
                [[lat, lon], [n_lat, n_lon]],
                color="#ffea00", weight=2, opacity=0.9,
                dash_array="5,5",
                tooltip=f"Nearest point on corridor axis "
                        f"({_fmt_km(dkm)} from you)",
            ).add_to(m)
            folium.CircleMarker(
                location=[n_lat, n_lon],
                radius=7, color="#000000", weight=3,
                fill=True, fill_color="#ffea00", fill_opacity=1.0,
                tooltip=f"{event['target']} transit "
                        f"({_fmt_km(dkm)} from you) at {event['time_utc']} UTC",
            ).add_to(m)

            folium.CircleMarker(
                [grid["sub_lat"], grid["sub_lon"]],
                radius=5, color="#222", weight=2,
                fill=True, fill_color="white", fill_opacity=1.0,
                tooltip=f"Sub-satellite point at {event['time_utc']} UTC",
            ).add_to(m)

            has_corridor = True
        except Exception as e:
            print(f"      tight grid failed: {e}", flush=True)

    folium.CircleMarker(
        location=[lat, lon], radius=8,
        color="#000000", weight=3,
        fill=True, fill_color="#ff0000", fill_opacity=1.0,
        popup="Observer (your position)",
        tooltip="Observer (your position)",
    ).add_to(m)

    folium.Circle(
        [lat, lon], radius=max_dist_km * 1000.0,
        color="#00aa00", weight=2, fill=False, dash_array="6,6",
        popup=f"Search radius: {max_dist_km:.0f} km",
    ).add_to(m)

    folium.CircleMarker(
        location=[lat, lon], radius=6,
        color="#000000", weight=3,
        fill=True, fill_color=color, fill_opacity=1.0,
        popup=(f"<b>{status}</b><br>"
               f"{event['sat']} → {event['target']}<br>"
               f"UTC: {event['time_utc']}<br>"
               f"Sep from your site: {event['sep_km']:.3f} km "
               f"({event.get('sep_deg', 0.0):.4f}°)<br>"
               f"Slant range: {event['sat_alt_km']:.0f} km"
               f"{condition_popup}"),
        tooltip=f"{status}: {event['sat']} → {event['target']}",
    ).add_to(m)

    _add_legend(
        m,
        confirmed_count=1 if status == "CONFIRMED" else 0,
        near_count=1 if status == "NEAR" else 0,
        tile_layer_name=tile_layer,
        has_corridor=has_corridor,
        physical_km=corridor_km,
        event_tags=condition_badges,
    )

    m.save(out_html)
    return out_html


# ═══════════════════════════════════════════════════════════════════
# Event classification
# ═══════════════════════════════════════════════════════════════════

def classify_event(event, sat_slant_km, target_name, earth, sat, target,
                   observer, ts, ephemeris, min_alt_deg=0.0, sat_name=None):
    t = ts.from_datetimes([event["time"].utc_datetime()])

    sun_alt, _, _ = ((earth + observer).at(t)
                     .observe(ephemeris["sun"]).apparent().altaz())
    daylight = float(np.atleast_1d(sun_alt.degrees)[0]) >= 0.0
    satellite_sunlit = bool(np.atleast_1d(sat.at(t).is_sunlit(ephemeris))[0])

    tgt_alt, _, _ = ((earth + observer).at(t)
                     .observe(target).apparent().altaz())
    tgt_alt_deg = float(np.atleast_1d(tgt_alt.degrees)[0])

    sat_alt_ao, _, _ = (sat - observer).at(t).altaz()
    sat_alt_deg = float(np.atleast_1d(sat_alt_ao.degrees)[0])

    corridor_km = disk_corridor_km(target_name, sat_slant_km,
                                   sat_name=sat_name)

    if tgt_alt_deg < min_alt_deg or sat_alt_deg < 0.0:
        return {
            "sub_lat": float("nan"),
            "sub_lon": float("nan"),
            "corridor_km": corridor_km,
            "sep_km": float("nan"),
            "sep_deg": float("nan"),
            "inside_corridor": False,
            "status": "REJECT",
            "daylight": daylight,
            "satellite_sunlit": satellite_sunlit,
            "reason": (f"below horizon "
                       f"(target_alt={tgt_alt_deg:.1f}°, "
                       f"sat_alt={sat_alt_deg:.1f}°)"),
        }

    sep_km, _, _, _, sep_deg = compute_separations(
        sat, target, earth, observer, t
    )
    sep_km = float(sep_km[0])
    sep_deg = float(sep_deg[0])

    inside = sep_km < corridor_km
    status = "CONFIRMED" if inside else "NEAR"

    sub = wgs84.subpoint_of(sat.at(t))
    sub_lat = float(np.atleast_1d(sub.latitude.degrees)[0])
    sub_lon = float(np.atleast_1d(sub.longitude.degrees)[0])

    return {
        "sub_lat": sub_lat,
        "sub_lon": sub_lon,
        "corridor_km": corridor_km,
        "sep_km": sep_km,
        "sep_deg": sep_deg,
        "inside_corridor": inside,
        "status": status,
        "daylight": daylight,
        "satellite_sunlit": satellite_sunlit,
        "reason": "",
    }


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    wall0 = time.time()
    print("═" * 72)
    print(" Space Station Transit Predictor — Verbose Edition")
    print("═" * 72)

    cfg = parse_config()
    print(f"\nObserver   : {cfg['lat']:.6f}, {cfg['lon']:.6f}  "
          f"({cfg['elev']:.0f} m)")
    print(f"Window     : {cfg['start']} → {cfg['end']}")
    print(f"Coarse     : {cfg['coarse_step_s']:.1f} s")
    print(f"Fine       : {cfg['fine_step_s']:.1f} s")
    print(f"Radius     : {cfg['max_dist_km']} km")
    print(f"Targets    : {len(cfg['targets'])}")
    print(f"Map dir    : {cfg['map_dir']}")
    print(f"Tile layers: {cfg['tile_layers']}")

    print("\n[1/5] Loading JPL ephemeris …")
    t0 = time.time()
    eph = load("de421.bsp")
    earth = eph["earth"]
    print(f"  Loaded de421.bsp in {time.time()-t0:.1f}s")

    observer = wgs84.latlon(cfg["lat"], cfg["lon"],
                            elevation_m=cfg["elev"])
    ts = load.timescale()

    print("\n[2/5] Loading satellites …")
    sats = load_satellites(cfg["tle_source"])
    print(f"  Found: {list(sats.keys())}")

    print("\n[3/5] Horizon pre-filter …")
    visible_targets = {}
    for name in cfg["targets"]:
        try:
            target = build_target(name, eph)
        except ValueError as e:
            print(f"  {name:12s} skipped: {e}")
            continue
        ever, peak = target_ever_visible(
            target, earth, observer,
            cfg["start"], cfg["end"], cfg["min_alt"],
        )
        if ever:
            print(f"  {name:12s} OK   peak altitude {peak:6.1f}°")
            visible_targets[name] = target
        else:
            print(f"  {name:12s} SKIP peak altitude {peak:6.1f}° "
                  f"(< {cfg['min_alt']}°)")

    if not visible_targets:
        print("\nNo targets rise above the minimum altitude.")
        return

    print("\n[4/5] Building shared coarse grid and caches …")
    coarse_times, n_coarse = build_coarse_grid(
        cfg["start"], cfg["end"], cfg["coarse_step_s"]
    )
    if coarse_times is None:
        print("  Coarse window too short; nothing to scan.")
        return
    print(f"  Coarse grid: {n_coarse} points at "
          f"{cfg['coarse_step_s']:.0f} s step")

    target_cache = cache_target_los(
        visible_targets, earth, observer, coarse_times,
        min_alt_deg=cfg["min_alt"],
    )
    sat_cache = cache_satellite_positions(sats, observer, coarse_times)

    print("\n[5/5] Scanning for transits …")
    all_events = []

    half_window_s = cfg["coarse_step_s"] * 1.5

    for sat_name, sat in sats.items():
        print(f"\n── {sat_name} ──")
        sat_entry = sat_cache[sat_name]

        for target_name, target in visible_targets.items():
            t_target = time.time()
            print(f"\n  → {target_name}", flush=True)

            tgt_entry = target_cache[target_name]
            coarse, n_hits = coarse_pair_scan(
                sat_entry, tgt_entry, coarse_times,
                cfg["max_dist_km"],
                earth=earth, observer=observer, sat=sat, target=target,
                min_alt_deg=cfg["min_alt"],
                above_horizon=(sat_entry["above_horizon"]
                               & tgt_entry["above_horizon"]),
                label=f"{sat_name}/{target_name}",
                verbose=True,
            )
            print(f"      [{sat_name}/{target_name}] coarse scan done "
                  f"({n_hits} hits), elapsed "
                  f"{time.time()-t_target:.1f}s", flush=True)

            if not coarse:
                continue

            fine = scan_fine(
                sat, target, earth, observer, coarse,
                cfg["fine_step_s"], half_window_s, cfg["max_dist_km"],
                label=f"{sat_name}/{target_name}",
                verbose=True,
                min_alt_deg=cfg["min_alt"],
            )
            if not fine:
                continue

            n_fine_raw = len(fine)
            fine = cluster_candidates(
                fine, cluster_gap_s=cfg["fine_step_s"] * 1.5
            )
            if n_fine_raw != len(fine):
                print(f"      clustered fine candidates: "
                      f"{n_fine_raw} → {len(fine)}", flush=True)

            sample_slant = fine[0]["sat_alt_km"]
            corridor_km = disk_corridor_km(target_name, sample_slant,
                                           sat_name=sat_name)
            print(f"      physical corridor = {_fmt_km(corridor_km)} "
                  f"(slant={sample_slant:.0f} km)", flush=True)

            target_events = []
            n_rejected = 0
            for i, c in enumerate(fine):
                t_pre = ts.from_datetimes([c["time"].utc_datetime()])
                tgt_alt, _, _ = ((earth + observer).at(t_pre)
                                 .observe(target).apparent().altaz())
                sat_alt_ao, _, _ = (sat - observer).at(t_pre).altaz()
                if (float(np.atleast_1d(tgt_alt.degrees)[0])
                        < cfg["min_alt"]
                        or float(np.atleast_1d(sat_alt_ao.degrees)[0]) < 0.0):
                    n_rejected += 1
                    continue

                try:
                    t_ref, sep_km, sep_deg = refine_transit(
                        sat, target, earth, observer, c["time"]
                    )
                except Exception as e:
                    print(f"      refine failed: {e}")
                    continue

                probe = {"time": t_ref,
                         "obs_lat": cfg["lat"],
                         "obs_lon": cfg["lon"]}
                cls = classify_event(
                    probe, c["sat_alt_km"], target_name,
                    earth, sat, target, observer, ts, eph,
                    min_alt_deg=cfg["min_alt"],
                    sat_name=sat_name,
                )

                if cls["status"] == "REJECT":
                    n_rejected += 1
                    print(f"        refine {i+1}/{len(fine)}: "
                          f"{t_ref.utc_strftime('%Y-%m-%d %H:%M:%S')}  "
                          f"[{colored_status(cls['status'])}] "
                          f"{cls['reason']}"
                          f"{format_event_condition_tags(cls)}",
                          flush=True)
                    continue

                print(f"        refine {i+1}/{len(fine)}: "
                      f"{t_ref.utc_strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}  "
                      f"sep={sep_deg:.4f}°  "
                      f"sep_perp={sep_km:.3f} km  "
                      f"[{colored_status(cls['status'])}]"
                      f"{format_event_condition_tags(cls)}", flush=True)

                target_events.append({
                    "sat": sat_name,
                    "target": target_name,
                    "time": t_ref,
                    "time_utc": t_ref.utc_strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                    "sep_km": sep_km,
                    "sep_deg": sep_deg,
                    "sat_alt_km": c["sat_alt_km"],
                    "status": cls["status"],
                    "sub_lat": cls["sub_lat"],
                    "sub_lon": cls["sub_lon"],
                    "corridor_km": cls["corridor_km"],
                    "inside_corridor": cls["inside_corridor"],
                    "daylight": cls["daylight"],
                    "satellite_sunlit": cls["satellite_sunlit"],
                    "obs_lat": cfg["lat"],
                    "obs_lon": cfg["lon"],
                })

            target_events = deduplicate(target_events, min_gap_s=30.0)
            n_conf = sum(1 for e in target_events
                         if e["status"] == "CONFIRMED")
            n_near = sum(1 for e in target_events
                         if e["status"] == "NEAR")
            event_word = "event" if len(target_events) == 1 else "events"
            print(f"      {len(target_events)} {event_word} after dedup "
                  f"({n_conf} CONFIRMED, {n_near} NEAR, "
                  f"{n_rejected} rejected), "
                  f"elapsed {time.time()-t_target:.1f}s", flush=True)

            all_events.extend(target_events)

    print("\n" + "═" * 72)
    confirmed = [e for e in all_events if e["status"] == "CONFIRMED"]
    near      = [e for e in all_events if e["status"] == "NEAR"]
    event_word = "event" if len(all_events) == 1 else "events"
    print(f"Total {event_word} in search radius: {len(all_events)} "
          f"({len(confirmed)} CONFIRMED, {len(near)} NEAR)")

    for e in sorted(all_events, key=lambda x: x["time"].utc_datetime()):
        print(f"  [{colored_status(e['status'])}] "
              f"{e['sat']:6s} → {e['target']:12s}  "
              f"{e['time_utc']} UTC   "
              f"sep={e.get('sep_deg', 0.0):7.4f}°   "
              f"sep_km={e['sep_km']:7.4f} km   "
              f"slant={e['sat_alt_km']:.0f} km"
              f"{format_event_condition_tags(e)}")

    target_objs = {name: build_target(name, eph)
                   for name in visible_targets}

    written = []
    for e in all_events:
        sat_obj = sats.get(e["sat"])
        tgt_obj = target_objs.get(e["target"])
        for tile in cfg["tile_layers"]:
            try:
                path = generate_transit_map(
                    e, cfg["lat"], cfg["lon"], cfg["max_dist_km"],
                    eph_earth=earth, sat=sat_obj, target=tgt_obj,
                    out_dir=cfg["map_dir"],
                    tile_layer=tile,
                )
                written.append(path)
            except Exception as e_map:
                print(f"  map failed for {e['sat']}→{e['target']} "
                      f"@ {e['time_utc']} [{tile}]: {e_map}")

    if written:
        print(f"\nWrote {len(written)} map(s) to {cfg['map_dir']}/:")
        for p in written:
            print(f"  {p}")

    print(f"\nTotal wall time: {time.time() - wall0:.1f}s")


if __name__ == "__main__":
    main()