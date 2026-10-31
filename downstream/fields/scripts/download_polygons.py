#!/usr/bin/env python3
"""FTW's field polygons for every country (kerner-lab/fields-of-the-world-<country>/boundaries_<country>_<year>.parquet
on Source Cooperative), the source of the labels.

    python scripts/download_polygons.py            # all 25 regions -> data/ftw_polygons/ (~0.5 GB)
"""

from __future__ import annotations

import argparse
from pathlib import Path

from select_chips import fetch_polygons

ROOT = Path(__file__).resolve().parent.parent
REGIONS = ["austria", "belgium", "brazil", "cambodia", "corsica", "croatia", "denmark", "estonia", "finland", "france",
           "germany", "india", "kenya", "latvia", "lithuania", "luxembourg", "netherlands", "portugal", "rwanda",
           "slovakia", "slovenia", "south_africa", "spain", "sweden", "vietnam"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--countries", nargs="+", default=REGIONS)
    args = ap.parse_args()
    folder = ROOT / "data/ftw_polygons"
    for c in args.countries:
        fetch_polygons(c, folder)                     # skipped if already there
        print(c, "ok", flush=True)


if __name__ == "__main__":
    main()
