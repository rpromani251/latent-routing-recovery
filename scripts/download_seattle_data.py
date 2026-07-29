#!/usr/bin/env python3
"""
Download the Seattle Building Energy Benchmarking Data (2015-Present) CSV into data/raw/.

The dataset lives on the Seattle Open Data portal (data.seattle.gov) and is updated
annually, so its export URL is not hardcoded here (a fabricated Socrata resource ID
would silently break rather than fail loudly). Pass the CSV export URL from the
dataset's "Export" panel, or set SEATTLE_EUI_CSV_URL:

    python scripts/download_seattle_data.py --url "https://data.seattle.gov/.../rows.csv"

If you already have the file (e.g. downloaded by hand from the portal), just place it at
data/raw/seattle_building_energy_benchmarking.csv directly -- this script is a
convenience, not a hard requirement.
"""
import argparse
import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.detect_recover_interpret import paths


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default=os.environ.get("SEATTLE_EUI_CSV_URL"),
                     help="Direct CSV export URL from data.seattle.gov")
    ap.add_argument("--out", type=Path, default=paths.SEATTLE_EUI_CSV)
    args = ap.parse_args()

    if not args.url:
        print(
            "No --url given and SEATTLE_EUI_CSV_URL is not set.\n\n"
            "Get the current export URL from the Seattle Open Data portal "
            "(search \"Building Energy Benchmarking\" at data.seattle.gov), "
            "then rerun with --url, or place the CSV yourself at:\n"
            f"  {args.out}",
            file=sys.stderr,
        )
        sys.exit(1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {args.url} -> {args.out}")
    urllib.request.urlretrieve(args.url, args.out)
    print(f"wrote {args.out}  ({args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
