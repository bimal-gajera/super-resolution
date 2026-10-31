#!/usr/bin/env python3
"""Which of our Sentinel-2 / Maxar training pairs lie on Fields of The World (FTW) label chips?

The pairs (../../Bimal/visual_small, 480 m tiles in UTM) are the only real sub-metre images we have. A Maxar tile
can serve as the real high-resolution reference of a field-boundary flow only where FTW has labels, so this lists
every pair whose footprint intersects an FTW chip.

    python scripts/maxar_overlap.py

Writes data/maxar_tiles.csv (one row per pair: key, EPSG code, UTM bounds, centre lon / lat; read once, then reused)
and data/maxar_on_ftw.csv (one row per intersecting pair and chip: key, country, aoi_id, split, the share of the
480 m tile inside the chip, and the distance between the centres in metres).
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import box

ROOT = Path(__file__).resolve().parent.parent
PAIRS = ROOT.parents[3] / "Bimal/visual_small"


def footprint(path: Path) -> dict:
    with rasterio.open(path) as src:
        b = src.bounds
        return dict(key=path.stem.replace("_S2", "_MAXAR"), epsg=src.crs.to_epsg(),
                    x_min=b.left, y_min=b.bottom, x_max=b.right, y_max=b.top)


def load_tiles(pairs: Path, cache: Path, threads: int) -> gpd.GeoDataFrame:
    if cache.exists():
        tiles = pd.read_csv(cache)
    else:
        paths = sorted((pairs / "Sentinel-2").glob("*.tif"))
        with ThreadPoolExecutor(threads) as pool:
            tiles = pd.DataFrame(pool.map(footprint, paths))
        parts = []
        for epsg, g in tiles.groupby("epsg"):          # centre in lon / lat, one projection per UTM zone
            c = gpd.GeoSeries(gpd.points_from_xy((g.x_min + g.x_max) / 2, (g.y_min + g.y_max) / 2),
                              crs=epsg, index=g.index).to_crs(4326)
            parts.append(g.assign(lon=c.x, lat=c.y))
        tiles = pd.concat(parts).sort_index()
        tiles.to_csv(cache, index=False)
        print(f"wrote {cache} ({len(tiles)} pairs)")
    parts = []
    for epsg, g in tiles.groupby("epsg"):
        geom = gpd.GeoSeries([box(*r) for r in g[["x_min", "y_min", "x_max", "y_max"]].to_numpy()],
                             crs=epsg, index=g.index).to_crs(4326)
        parts.append(gpd.GeoDataFrame(g, geometry=geom, crs=4326))
    return pd.concat(parts).sort_index()


def load_chips(folder: Path) -> gpd.GeoDataFrame:
    frames = []
    for path in sorted(folder.glob("chips_*.parquet")):
        chips = gpd.read_parquet(path).to_crs(4326)
        chips["aoi_id"] = chips["aoi_id"].astype(str)
        frames.append(chips.assign(country=path.stem[len("chips_"):]))
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", type=Path, default=PAIRS)
    ap.add_argument("--chips", type=Path, default=ROOT / "data/ftw_chips")
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()

    tiles = load_tiles(args.pairs, ROOT / "data/maxar_tiles.csv", args.threads)
    chips = load_chips(args.chips)
    print(f"{len(tiles)} pairs in {tiles.epsg.nunique()} UTM zones; {len(chips)} FTW chips in "
          f"{chips.country.nunique()} regions")

    # coarse picture first: how far is each pair from the nearest FTW chip (degrees, on chip and tile centres)?
    b = chips.geometry.bounds
    cc = np.column_stack([(b.minx + b.maxx) / 2, (b.miny + b.maxy) / 2])
    lo, hi = cc.min(0) - 1, cc.max(0) + 1
    near = tiles[(tiles.lon.between(lo[0], hi[0])) & (tiles.lat.between(lo[1], hi[1]))]
    print(f"pairs inside the bounding box of all FTW chips (+-1 deg): {len(near)}")
    for country, g in chips.groupby("country"):
        b = g.total_bounds
        n = int((tiles.lon.between(b[0], b[2]) & tiles.lat.between(b[1], b[3])).sum())
        if n:
            print(f"  inside the bounding box of {country}: {n} pairs")

    hits = gpd.sjoin(tiles, chips, predicate="intersects").rename(columns={"index_right": "chip"})
    if hits.empty:
        print("no pair intersects an FTW chip")
        return
    share, dist = [], []
    for row in hits.itertuples():
        chip = chips.geometry.iloc[row.chip]
        utm = gpd.GeoSeries([row.geometry, chip], crs=4326).to_crs(int(row.epsg))
        share.append(utm.iloc[0].intersection(utm.iloc[1]).area / utm.iloc[0].area)
        dist.append(utm.iloc[0].centroid.distance(utm.iloc[1].centroid))
    hits = hits.assign(share_in_chip=np.round(share, 3), centre_dist_m=np.round(dist, 1))
    out = ROOT / "data/maxar_on_ftw.csv"
    hits[["key", "country", "aoi_id", "split", "share_in_chip", "centre_dist_m", "lon", "lat"]].to_csv(out, index=False)
    print(f"wrote {out}: {hits.key.nunique()} pairs on {hits.groupby(['country', 'aoi_id']).ngroups} chips")
    print(hits.groupby(["country", "split"]).agg(pairs=("key", "nunique"), chips=("aoi_id", "nunique"),
                                                 share_median=("share_in_chip", "median")).to_string())


if __name__ == "__main__":
    main()
