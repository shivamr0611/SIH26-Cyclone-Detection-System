#!/usr/bin/env python3
"""
download_ibtracs.py
Downloads the official NOAA IBTrACS North Indian Ocean (NIO) best-track dataset
and saves it directly into database/ibtracs/ibtracs_NIO.csv.
"""

import argparse
import sys
import urllib.request
from pathlib import Path

# Base paths relative to repository root
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DESTINATION = REPO_ROOT / "database" / "ibtracs" / "ibtracs_NIO.csv"

# NOAA IBTrACS North Indian Ocean (NIO) direct CSV URL
NOAA_IBTRACS_NIO_URL = (
    "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/"
    "v04r00/access/csv/ibtracs.NI.list.v04r00.csv"
)


def download_progress_hook(block_num: int, block_size: int, total_size: int) -> None:
    """Print download progress in percentage and MB."""
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100.0, downloaded * 100.0 / total_size)
        sys.stdout.write(
            f"\rDownloading: {percent:5.1f}% ({downloaded / (1024 * 1024):.2f} MB / {total_size / (1024 * 1024):.2f} MB)"
        )
    else:
        sys.stdout.write(f"\rDownloading: {downloaded / (1024 * 1024):.2f} MB")
    sys.stdout.flush()


def download_ibtracs_nio(url: str, output_path: Path) -> bool:
    """Download IBTrACS NIO CSV to target destination."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Fetching IBTrACS NIO dataset from:\n  {url}")
    print(f"Destination:\n  {output_path}")

    try:
        urllib.request.urlretrieve(url, str(output_path), reporthook=download_progress_hook)
        print(f"\n[SUCCESS] Download complete: {output_path} ({output_path.stat().st_size // 1024} KB)")
        return True
    except Exception as err:
        print(f"\n[ERROR] Failed to download IBTrACS dataset: {err}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Download official NOAA IBTrACS NIO dataset.")
    parser.add_argument(
        "--url",
        type=str,
        default=NOAA_IBTRACS_NIO_URL,
        help="Custom URL for IBTrACS NIO CSV",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_DESTINATION,
        help=f"Target path for CSV (default: {DEFAULT_DESTINATION})",
    )
    args = parser.parse_args()

    download_ibtracs_nio(args.url, args.output)


if __name__ == "__main__":
    main()
