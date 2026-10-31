#!/usr/bin/env python3
"""Download a sample of Fields of The World (FTW) test chips and pick the chips of a check.

    python scripts/select_chips.py --countries belgium lithuania latvia croatia cambodia south_africa

Per country: a random sample of test chips (seeded) is fetched file by file from Source Cooperative into
data/ftw/<country>/ in FTW's own layout (s2_images/window_{a,b}, label_masks/{instance,semantic_3class}), plus the
country's polygon file into data/ftw_polygons/. Every sampled chip gets a row in data/chip_stats.csv (cloud and
nodata share per date, number of fields, median field area, labelled share). The selection keeps chips that are
clear on both dates and have enough fields, and spreads them from small to large median field size:

    data/selections/<name>.txt           `<country> <aoi_id>` per line
    data/selections/<name>_one_each.txt  one chip per country (the median one), for variants run on fewer chips
"""

from __future__ import annotations

import argparse
import re
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

from ftw_geo import ROOT, WINDOWS, chip_path

BASE = "https://s3.us-west-2.amazonaws.com/us-west-2.opendata.source.coop"
KINDS = (*WINDOWS, "instance", "semantic_3class")
CLOUD_B02 = 2200      # blue above this DN (reflectance 0.22) is cloud, snow or thick haze
HAZE_B02 = 1400       # median blue above this: hazy scene


def fetch(url: str, path: Path) -> bool:
    """False if the file does not exist on the server (some chips of the split tables have no image or label)."""
    if path.exists() and path.stat().st_size:
        return True
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{threading.get_ident()}.part")
    try:
        urllib.request.urlretrieve(url, tmp)
    except urllib.error.HTTPError as err:
        if err.code in (403, 404):
            return False
        raise
    tmp.replace(path)
    return True


def fetch_polygons(country: str, folder: Path) -> None:
    if list(folder.glob(f"boundaries_{country}_*.parquet")):
        return
    prefix = f"kerner-lab/fields-of-the-world-{country.replace('_', '')}/"
    listing = urllib.request.urlopen(f"{BASE}?list-type=2&prefix={prefix}").read().decode()
    key = next(k for k in re.findall(r"<Key>(.*?)</Key>", listing) if k.endswith(".parquet"))
    fetch(f"{BASE}/{key}", folder / Path(key).name)


def chip_stats(ftw: Path, country: str, aoi_id: str) -> dict:
    row = dict(country=country, aoi_id=aoi_id)
    images = {}
    for kind in WINDOWS:
        with rasterio.open(chip_path(ftw, country, aoi_id, kind)) as src:
            images[kind] = src.read()
            lat = src.transform.f + src.transform.e * src.height / 2
            px_area = abs(src.transform.a * 111320 * np.cos(np.radians(lat)) * src.transform.e * 110574)
    # Chips of 2022 and later carry the +1000 DN offset of Sentinel-2 L2A (no dark pixel below 1000), older ones
    # do not; the cloud and haze thresholds refer to reflectance, so the offset is taken off first.
    row["offset"] = 1000 if min(int(im[:3].min()) for im in images.values()) >= 950 else 0
    for kind, image in images.items():
        blue = image[2].astype(np.int32) - row["offset"]
        row[f"cloud_{kind[-1]}"] = float((blue > CLOUD_B02).mean())
        row[f"nodata_{kind[-1]}"] = float((image == 0).all(0).mean())
        row[f"b02_median_{kind[-1]}"] = float(np.median(blue))
    with rasterio.open(chip_path(ftw, country, aoi_id, "instance")) as src:
        ids, counts = np.unique(src.read(1), return_counts=True)
    areas = counts[ids != 0] * px_area / 1e4                     # hectares
    with rasterio.open(chip_path(ftw, country, aoi_id, "semantic_3class")) as src:
        row["labelled"] = float((src.read(1) > 0).mean())
    row.update(n_fields=len(areas), field_ha_median=float(np.median(areas)) if len(areas) else np.nan,
               small_fields=float((areas < 0.5).mean()) if len(areas) else np.nan)
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--countries", nargs="+", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--sample", type=int, default=30, help="chips downloaded per country")
    ap.add_argument("--per-country", type=int, default=5, help="chips selected per country")
    ap.add_argument("--name", default="first_check")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-fields", type=int, default=8)
    ap.add_argument("--min-labelled", type=float, default=0.25)
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()
    ftw = ROOT / "data/ftw"

    jobs, sampled = [], []
    for country in args.countries:
        fetch_polygons(country, ROOT / "data/ftw_polygons")
        chips = gpd.read_parquet(ROOT / f"data/ftw_chips/chips_{country}.parquet")
        ids = sorted(chips[chips.split == args.split].aoi_id.astype(str))
        ids = list(np.random.default_rng(args.seed).permutation(ids)[: args.sample])
        sampled += [(country, i) for i in ids]
        for i in ids:
            for kind in KINDS:
                path = chip_path(ftw, country, i, kind)
                jobs.append((f"{BASE}/kerner-lab/fields-of-the-world/{path.relative_to(ftw)}", path))
    with ThreadPoolExecutor(args.threads) as pool:
        list(pool.map(lambda j: fetch(*j), jobs))
    complete = [(c, i) for c, i in sampled if all(chip_path(ftw, c, i, k).exists() for k in KINDS)]
    print(f"{len(complete)} of {len(sampled)} sampled chips have both dates and labels ({ftw})")
    sampled = complete

    stats = pd.DataFrame([chip_stats(ftw, c, i) for c, i in sampled])
    worst = lambda name: stats[[f"{name}_a", f"{name}_b"]].max(axis=1)  # noqa: E731
    stats["clear"] = (worst("cloud") < 0.005) & (worst("nodata") < 0.005) & (worst("b02_median") < HAZE_B02)
    stats["usable"] = stats.clear & (stats.n_fields >= args.min_fields) & (stats.labelled >= args.min_labelled)
    stats.to_csv(ROOT / "data/chip_stats.csv", index=False)

    chosen, one_each = [], []
    for country in args.countries:
        g = stats[(stats.country == country) & stats.usable].sort_values(["field_ha_median", "aoi_id"])
        n = min(args.per_country, len(g))
        picks = g.iloc[[int((k + 0.5) / n * len(g)) for k in range(n)]] if n else g
        chosen.append(picks)
        if n:
            one_each.append(picks.iloc[[n // 2]])
        print(f"{country}: {len(g)} of {(stats.country == country).sum()} sampled chips usable; picked {n}")
    chosen, one_each = pd.concat(chosen), pd.concat(one_each)
    out = ROOT / "data/selections"
    out.mkdir(parents=True, exist_ok=True)
    for frame, name in ((chosen, args.name), (one_each, f"{args.name}_one_each")):
        (out / f"{name}.txt").write_text("".join(f"{r.country} {r.aoi_id}\n" for r in frame.itertuples()))
    print(chosen[["country", "aoi_id", "offset", "n_fields", "field_ha_median", "small_fields", "labelled"]]
          .round(2).to_string(index=False))


if __name__ == "__main__":
    main()
