#!/usr/bin/env python3
"""
label_from_ibtracs.py
Extracts cyclone tracks and intensity from IBTrACS NIO dataset,
classifies wind speeds according to IMD categories, and produces
image_labels.json metadata mapping.
"""

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Base paths relative to repository root
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_IBTRACS_CSV = REPO_ROOT / "database" / "ibtracs" / "ibtracs_NIO.csv"
DEFAULT_OUTPUT_JSON = REPO_ROOT / "database" / "metadata" / "image_labels.json"
DEFAULT_RAW_DIR = REPO_ROOT / "database" / "raw"


def get_imd_category(wind_kt: float) -> str:
    """Map maximum sustained wind speed (knots) to IMD Category.

    IMD Criteria:
      < 28 kt   : Depression
      28-33 kt  : Deep_Depression
      34-47 kt  : Cyclonic_Storm
      48-63 kt  : Severe_Cyclonic_Storm
      >= 64 kt  : Very_Severe_Cyclonic_Storm
    """
    if wind_kt < 28.0:
        return "Depression"
    elif 28.0 <= wind_kt <= 33.0:
        return "Deep_Depression"
    elif 34.0 <= wind_kt <= 47.0:
        return "Cyclonic_Storm"
    elif 48.0 <= wind_kt <= 63.0:
        return "Severe_Cyclonic_Storm"
    else:
        return "Very_Severe_Cyclonic_Storm"


def _extract_val(row: Dict[str, str], candidates: list[str]) -> Optional[str]:
    """Find first matching column candidate in row dict (case-insensitive)."""
    normalized_row = {k.strip().lower(): v.strip() for k, v in row.items() if k}
    for cand in candidates:
        cand_lower = cand.lower()
        if cand_lower in normalized_row and normalized_row[cand_lower]:
            val = normalized_row[cand_lower]
            if val not in ("-999", "-999.0", "MISSING", "NA", "null", ""):
                return val
    return None


def parse_ibtracs_row(row: Dict[str, str]) -> Optional[Dict[str, Any]]:
    """Parse IBTrACS row extracting storm name, timestamp, wind_kt, mslp, lat, lon."""
    storm_name = _extract_val(row, ["NAME", "storm_name", "storm", "STORM_NAME"]) or "UNNAMED"
    iso_time = _extract_val(row, ["ISO_TIME", "timestamp", "time", "datetime", "DATE_TIME"])
    if not iso_time:
        return None

    # Wind speed in knots (check WMO_WIND, USA_WIND, NEWDELHI_WIND, wind_kt)
    wind_str = _extract_val(
        row, ["WMO_WIND", "USA_WIND", "NEWDELHI_WIND", "REUNION_WIND", "wind_kt", "wind", "WIND_KTS"]
    )
    if wind_str is None:
        return None
    try:
        wind_kt = float(wind_str)
    except ValueError:
        return None

    # Central pressure in hPa / mb
    mslp_str = _extract_val(
        row, ["WMO_PRES", "USA_PRES", "NEWDELHI_PRES", "REUNION_PRES", "mslp", "pres", "pressure"]
    )
    mslp: Optional[float] = None
    if mslp_str:
        try:
            mslp = float(mslp_str)
        except ValueError:
            mslp = None

    # Latitude and Longitude
    lat_str = _extract_val(row, ["LAT", "lat", "latitude", "LATITUDE"])
    lon_str = _extract_val(row, ["LON", "lon", "longitude", "LONGITUDE"])
    if not lat_str or not lon_str:
        return None
    try:
        lat = float(lat_str)
        lon = float(lon_str)
    except ValueError:
        return None

    category = get_imd_category(wind_kt)

    # Format standard clean filename key
    clean_time = re.sub(r"[^\w\d]", "_", iso_time.strip())
    clean_storm = re.sub(r"[^\w\d]", "_", storm_name.strip().upper())
    filename = f"{clean_storm}_{clean_time}.png"

    return {
        "filename": filename,
        "metadata": {
            "category": category,
            "wind_kt": wind_kt,
            "mslp": mslp,
            "lat": lat,
            "lon": lon,
            "storm_name": storm_name,
            "timestamp": iso_time,
        },
    }


def process_ibtracs_csv(csv_path: Path) -> Dict[str, Dict[str, Any]]:
    """Read IBTrACS CSV and return mapping of {filename: metadata}."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"IBTrACS CSV file not found at: {csv_path}\n"
            f"Please download the NIO CSV from NOAA IBTrACS into database/ibtracs/ibtracs_NIO.csv"
        )

    labels_map: Dict[str, Dict[str, Any]] = {}
    with open(csv_path, mode="r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            return labels_map

        # Official IBTrACS has a secondary unit row (e.g. 'kts', 'mb', 'degrees_north')
        first_data = next(reader, None)
        if not first_data:
            return labels_map

        # Check if first_data looks like units row
        is_unit_row = any(
            unit_hint in (v or "").lower()
            for v in first_data
            for unit_hint in ("kts", "mb", "degrees_north", "yyyy-mm-dd", "iso")
        )

        f.seek(0)
        dict_reader = csv.DictReader(f)
        if is_unit_row:
            # Skip header and unit row
            next(dict_reader, None)

        for row in dict_reader:
            parsed = parse_ibtracs_row(row)
            if parsed:
                labels_map[parsed["filename"]] = parsed["metadata"]

    return labels_map


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract storm intensity labels from IBTrACS NIO dataset and build metadata."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_IBTRACS_CSV,
        help=f"Path to IBTrACS NIO CSV (default: {DEFAULT_IBTRACS_CSV})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
        help=f"Path to output JSON (default: {DEFAULT_OUTPUT_JSON})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print summary and sample mappings to stdout without writing files.",
    )
    args = parser.parse_args()

    print(f"Reading IBTrACS records from: {args.csv}")
    try:
        labels_map = process_ibtracs_csv(args.csv)
    except FileNotFoundError as err:
        print(f"[ERROR] {err}")
        return

    print(f"Successfully processed {len(labels_map)} storm records.")

    # Category counts summary
    category_counts: Dict[str, int] = {}
    for meta in labels_map.values():
        cat = meta["category"]
        category_counts[cat] = category_counts.get(cat, 0) + 1

    print("\nIMD Category Distribution:")
    for cat, count in sorted(category_counts.items(), key=lambda x: x[0]):
        print(f"  - {cat:30s}: {count:5d}")

    if args.dry_run:
        print("\n--- DRY RUN MODE: Sample 5 Mappings ---")
        sample_keys = list(labels_map.keys())[:5]
        sample_dict = {k: labels_map[k] for k in sample_keys}
        print(json.dumps(sample_dict, indent=2))
        print(f"\n[DRY RUN] Would write {len(labels_map)} items to {args.output}")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(labels_map, f, indent=2)
        print(f"\nSaved {len(labels_map)} label mappings to: {args.output}")


if __name__ == "__main__":
    main()
