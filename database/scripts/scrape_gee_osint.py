#!/usr/bin/env python3
"""
scrape_gee_osint.py
Pulls MODIS/Terra thermal infrared imagery via Google Earth Engine (GEE),
clips to the Bay of Bengal / North Indian Ocean region (lon 60-100, lat 5-25),
and exports GeoTIFFs to database/raw/insat3d_tir/.

Inspired by mserman90/osint-satellite-toolkit & Bellingcat RS4OSINT methodology.
"""

import argparse
import os
import sys
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Optional

# Base paths relative to repository root
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "database" / "raw" / "insat3d_tir"

# Bay of Bengal / North Indian Ocean bounding box: [minLon, minLat, maxLon, maxLat]
BOB_BBOX = [60.0, 5.0, 100.0, 25.0]


def print_gee_auth_help() -> None:
    """Print instructions on configuring Google Earth Engine credentials."""
    print(
        """
================================================================================
                    Google Earth Engine Authentication Guide
================================================================================
To access GEE datasets via the Python API, you must authenticate once:

1. Install the GEE CLI (included with earthengine-api):
   $ pip install earthengine-api

2. Run the authentication command in your terminal:
   $ earthengine authenticate

3. Follow the web flow to log in with your Google account authorized for GEE,
   and paste the authorization token back into the terminal.

4. If you have a Google Cloud Project for GEE, pass it via:
   $ python database/scripts/scrape_gee_osint.py --project YOUR_GCP_PROJECT_ID ...
   or set the EARTHENGINE_PROJECT environment variable.
================================================================================
"""
    )


def init_earth_engine(project_id: Optional[str] = None) -> bool:
    """Initialize Earth Engine API with optional project ID."""
    try:
        import ee  # type: ignore
    except ImportError:
        print("[ERROR] earthengine-api is not installed. Install it via: pip install earthengine-api")
        return False

    project = project_id or os.getenv("EARTHENGINE_PROJECT")
    try:
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()
        print(f"[SUCCESS] Earth Engine initialized successfully (Project: {project or 'Default'}).")
        return True
    except Exception as err:
        print(f"[ERROR] Failed to initialize Google Earth Engine: {err}")
        print_gee_auth_help()
        return False


def fetch_modis_thermal_imagery(
    storm_name: str,
    start_date: str,
    end_date: str,
    output_dir: Path,
    band: str = "LST_Day_1km",
    scale: int = 2000,
    project_id: Optional[str] = None,
) -> None:
    """Query MODIS Terra imagery for the given date range and bounding box, and save GeoTIFFs."""
    import ee  # type: ignore

    if not init_earth_engine(project_id):
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    region = ee.Geometry.Rectangle(BOB_BBOX)

    print(f"\nSearching MODIS/Terra dataset for Storm: '{storm_name.upper()}'")
    print(f"Date Range: {start_date} to {end_date}")
    print(f"Bounding Box: Lon [{BOB_BBOX[0]}, {BOB_BBOX[2]}], Lat [{BOB_BBOX[1]}, {BOB_BBOX[3]}]")

    # MODIS/061/MOD11A1 (Terra Land Surface Temperature / Band 31 Thermal Infrared 10.8um Proxy)
    collection = (
        ee.ImageCollection("MODIS/061/MOD11A1")
        .filterDate(start_date, end_date)
        .filterBounds(region)
        .select(band)
    )

    image_count = collection.size().getInfo()
    print(f"Found {image_count} scenes in collection.")

    if image_count == 0:
        print("[WARNING] No scenes found for specified dates. Try expanding the date range.")
        return

    image_list = collection.toList(image_count)
    for idx in range(image_count):
        image = ee.Image(image_list.get(idx)).clip(region)
        img_info = image.getInfo()
        img_id = img_info.get("id", f"scene_{idx}")
        date_str = img_id.split("/")[-1] if "/" in img_id else f"scene_{idx}"

        clean_storm = storm_name.replace(" ", "_").upper()
        out_filename = f"{clean_storm}_{date_str}_{band}.tif"
        out_path = output_dir / out_filename

        print(f"[{idx + 1}/{image_count}] Generating download URL for: {out_filename}")
        try:
            download_url = image.getDownloadURL(
                {
                    "name": out_filename,
                    "scale": scale,
                    "crs": "EPSG:4326",
                    "region": region.getInfo()["coordinates"],
                    "format": "GEO_TIFF",
                }
            )

            print(f"    Downloading to: {out_path} ...")
            urllib.request.urlretrieve(download_url, str(out_path))
            print(f"    [SAVED] {out_path.name} ({out_path.stat().st_size // 1024} KB)")
        except Exception as e:
            print(f"    [ERROR] Download failed for {out_filename}: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape MODIS/Terra thermal satellite imagery from Google Earth Engine for cyclone analysis."
    )
    parser.add_argument(
        "--storm-name",
        type=str,
        default="Cyclone_Fani",
        help="Cyclone name (e.g. 'Cyclone_Fani', 'Amphan', 'Mocha')",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2019-05-01",
        help="Start date in YYYY-MM-DD format (default: 2019-05-01)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2019-05-04",
        help="End date in YYYY-MM-DD format (default: 2019-05-04)",
    )
    parser.add_argument(
        "--band",
        type=str,
        default="LST_Day_1km",
        help="Band to extract from MODIS/061/MOD11A1 (e.g. LST_Day_1km, Emis_31, LST_Night_1km)",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=2500,
        help="Spatial resolution scale in meters for export (default: 2500)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save downloaded GeoTIFFs (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="Optional Google Cloud Project ID for GEE initialization",
    )
    parser.add_argument(
        "--auth-help",
        action="store_true",
        help="Display GEE setup and authentication instructions and exit.",
    )
    args = parser.parse_args()

    if args.auth_help:
        print_gee_auth_help()
        return

    fetch_modis_thermal_imagery(
        storm_name=args.storm_name,
        start_date=args.start_date,
        end_date=args.end_date,
        output_dir=args.output_dir,
        band=args.band,
        scale=args.scale,
        project_id=args.project,
    )


if __name__ == "__main__":
    main()
