# Transit Predictor

Transit Predictor searches for apparent alignments between the International
Space Station (ISS) or China's Tiangong space station and selected Solar System
objects or bright stars. It classifies events as `CONFIRMED`, `NEAR`, or
`REJECT`, and can generate interactive HTML maps showing the predicted ground
corridor around the observer.

## Features

- Coarse-to-fine event search with sub-second transit refinement.
- Observer-specific event classification and physical corridor widths.
- Interactive Folium maps with the observer, corridor, nearest point on its
  centerline, and satellite subpoint.
- Configurable observer, search window, targets, TLE source, and map tiles.
- Solar System targets: Moon, Sun, Venus, Mars, Jupiter, Saturn, Uranus, and
  Neptune.
- Proper-named bright stars down to visual magnitude 2.0, including Polaris.
  Their J2000 coordinates are from HYG Database v4.1.

## Requirements

- Python 3.10 or newer
- Internet access to retrieve TLEs and map tiles
- Dependencies in `requirements.txt`
- The JPL DE421 ephemeris file; Skyfield downloads it on first use if it is not
  already available

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp config.example.ini config.ini
```

Edit `config.ini` and set the observer's latitude, longitude, and elevation,
the search end date, and the desired targets. `config.ini` is intentionally
ignored by Git because it can contain a private observer location.

Run the predictor from the project directory:

```bash
python transit.py
```

The program fetches current station TLEs from Celestrak by default. It stores
generated HTML maps in the configured output directory (`transit_maps/` by
default). Map tiles are provided by the selected tile service when the map is
viewed, so an internet connection is needed to display the basemap.

## Configuration

See [`config.example.ini`](config.example.ini) for all supported sections and
options.

- `Observer`: latitude, longitude, and elevation in metres.
- `Search`: maximum distance in kilometres, dates, scan steps, and minimum
  target altitude.
- `Targets`: comma-separated supported object names. Keep this list selective
  because adding targets increases search time.
- `Satellites`: Celestrak TLE URL.
- `Output`: map directory and one or more supported tile layers.

Search dates and generated event timestamps are interpreted as UTC. If
`start_date` is empty or omitted, the search starts at the current UTC date and
time. `end_date` is required and uses `YYYY-MM-DD`; it is an exclusive endpoint
at midnight UTC at the start of that date.

Supported tile layers: `esri_imagery`, `esri_street`, `esri_topo`, and `osm`.
Multiple layers can be selected with `tile_layers = esri_imagery, osm`.

## Tests

Install the development dependencies and run the suite:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

## Accuracy notes

Results depend on the age and quality of the station TLEs. TLEs are predictions
and become less accurate as they age, especially for low-Earth-orbit objects.
The corridor is a geometric prediction based on the supplied ephemeris, TLE,
observer coordinates, and target position; it is not a guarantee of naked-eye
visibility. Check that the satellite and target are above the horizon for the
observer and event time.

The star coordinates are fixed at J2000 and do not apply proper-motion updates.
Solar System angular radii and satellite physical radii are approximate model
inputs used to size the displayed corridor.

## Data and attribution

Bright-star positions are from the [HYG Database v4.1], licensed under
[CC BY-SA 4.0]. The JPL DE421 ephemeris and station TLEs are separate data
products retrieved by Skyfield and Celestrak, respectively.

[HYG Database v4.1]: https://github.com/astronexus/HYG-Database
[CC BY-SA 4.0]: https://creativecommons.org/licenses/by-sa/4.0/

## License

This project is licensed under the MIT License; see [`LICENSE`](LICENSE).
