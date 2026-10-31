#!/usr/bin/env python3
"""Write FTW chips in the dataset layout of the Fields of the Planet code, with a flow's 3 m images in place of
PlanetScope, so that the paper's training and evaluation code runs on them unchanged.

    python scripts/make_ftp_layout.py --flow sr03 --train data/selections/a.txt --val data/selections/b.txt \
        --test data/selections/c.txt

Layout under data/ftp_layout/<name> (what ftw_planet.datasets.FTWPlanet reads):
    planet/index.parquet                          one row per chip: patch_id, country, usable_pair, image paths
    planet/<country>/window_{a,b}/<aoi_id>.tif    4 bands (R, G, B, NIR), uint16, 3 m, on the chip's UTM grid
    planet/<country>/labels/<aoi_id>.tif          0 background, 1 field, 2 boundary (uint8)
    ftw/<country>/chips_<country>.parquet         aoi_id and split (the split table the code joins on); a chip
                                                  listed in several of --train / --val / --test gets one row each

Flows:
    sr03     our 0.3 m PixelDiT (RGB) mosaic (scripts/mosaic.py, 1 m) averaged to 3 m; NIR from Sentinel-2
    bicubic  Sentinel-2 R, G, B upsampled bicubically to 3 m; NIR the same way
NIR is Sentinel-2 B08 upsampled bicubically in both flows, so the paper's model keeps its 8 input channels.

Values: the code divides by 10,000 (PlanetScope reflectance). Sentinel-2 bands are written as reflectance x 10,000
(the +1000 DN offset of 2022+ chips taken off); the SR's 8-bit RGB as value x 10,000 / 255, so it lands in 0-1.
Labels follow FTP: the FTW polygons clipped to the chip and rasterised on the 3 m grid, the boundary class drawn by
buffering each polygon's outline by 3 m.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import torch
from rasterio.features import rasterize
from scipy import ndimage
from shapely.geometry import box

from ftw_geo import (ROOT, WINDOWS, block_mean, chip_footprint, chip_grid, chip_offset, chip_path, load_polygons,
                     read_lr, read_selection)

RES = 3            # metres
BOUNDARY_M = 3.0   # buffer around each outline that becomes the boundary class (FTP's choice)


def to_1m(lr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """(bands, h, w) on the 10 m grid -> (H, W, bands) at 1 m, bicubic."""
    x = torch.nn.functional.interpolate(torch.from_numpy(lr[None]), size=shape, mode="bicubic", align_corners=False)
    return x[0].permute(1, 2, 0).numpy()


def fill_uncovered(mosaic: np.ndarray) -> np.ndarray:
    """Pixels no SR window reached (a strip of a few metres along the grid edge, 0 in all bands) take the value of
    the nearest covered pixel, so the downstream model does not see an artificial black border."""
    covered = mosaic.any(2)
    if covered.all():
        return mosaic
    idx = ndimage.distance_transform_edt(~covered, return_distances=False, return_indices=True)
    return mosaic[idx[0], idx[1]]


def labels(polys: gpd.GeoSeries, footprint, shape: tuple[int, int], transform) -> np.ndarray:
    clipped = [g for g in polys.intersection(footprint) if not g.is_empty and g.area > 0]
    out = np.zeros(shape, np.uint8)
    if not clipped:
        return out
    rasterize(clipped, out=out, transform=transform, default_value=1)
    rings = []
    for g in clipped:
        for part in getattr(g, "geoms", [g]):
            if part.geom_type == "Polygon":
                rings.append(part.exterior.buffer(BOUNDARY_M))
    if rings:
        rasterize(rings, out=out, transform=transform, default_value=2)
    return out


def write(path: Path, array: np.ndarray, transform, epsg: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bands = array[None] if array.ndim == 2 else array.transpose(2, 0, 1)
    with rasterio.open(path, "w", driver="GTiff", width=bands.shape[2], height=bands.shape[1], count=bands.shape[0],
                       dtype=bands.dtype, crs=f"EPSG:{epsg}", transform=transform, compress="deflate") as dst:
        dst.write(bands)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--flow", required=True, choices=("sr03", "bicubic"))
    ap.add_argument("--train", type=Path, default=None)
    ap.add_argument("--val", type=Path, default=None)
    ap.add_argument("--test", type=Path, default=None)
    ap.add_argument("--name", default=None, help="layout folder name (default: the flow)")
    ap.add_argument("--ftw", type=Path, default=ROOT / "data/ftw")
    ap.add_argument("--polygons", type=Path, default=ROOT / "data/ftw_polygons")
    ap.add_argument("--mosaic", type=Path, default=ROOT / "data/mosaic/pixeldit_rgb")
    args = ap.parse_args()
    out = ROOT / "data/ftp_layout" / (args.name or args.flow)

    splits: dict[tuple[str, str], list[str]] = {}
    for split in ("train", "val", "test"):
        sel = getattr(args, split)
        for chip in read_selection(sel) if sel else []:
            splits.setdefault(chip, []).append(split)
    if not splits:
        ap.error("give at least one of --train / --val / --test")

    index, tables = [], {}
    for (country, aoi_id), chip_splits in splits.items():
        first = chip_path(args.ftw, country, aoi_id, WINDOWS[0])
        grid = chip_grid(first)
        h1, w1 = grid.shape(1.0)
        shape3, transform3 = (h1 // RES, w1 // RES), grid.transform(RES)
        offset = chip_offset(args.ftw, country, aoi_id)
        west, south, east, north = chip_footprint(first)
        footprint = gpd.GeoSeries([box(west, south, east, north)], crs=4326).to_crs(grid.epsg).iloc[0]
        paths = {}
        for window in WINDOWS:
            lr, _ = read_lr(chip_path(args.ftw, country, aoi_id, window), grid, bands=(1, 2, 3, 4))
            s2 = np.clip(block_mean(to_1m(lr - offset, (h1, w1)), RES), 0, 65535)
            if args.flow == "sr03":
                mosaic = np.load(args.mosaic / country / f"{aoi_id}_{window}.npy")
                rgb = block_mean(fill_uncovered(mosaic).astype(np.float32), RES) * (10000 / 255)
                image = np.concatenate([rgb, s2[..., 3:4]], axis=2)
            else:
                image = s2
            rel = f"{country}/{window}/{aoi_id}.tif"
            write(out / "planet" / rel, np.round(image).astype(np.uint16), transform3, grid.epsg)
            paths[window] = rel
        lab = f"{country}/labels/{aoi_id}.tif"
        write(out / "planet" / lab, labels(load_polygons(args.polygons, country, grid), footprint, shape3, transform3),
              transform3, grid.epsg)
        geom = box(west, south, east, north)
        index.append(dict(patch_id=aoi_id, country=country, usable_pair=True, image_a_path=paths["window_a"],
                          image_b_path=paths["window_b"], label_path=lab, crs=f"EPSG:{grid.epsg}", geometry=geom))
        tables.setdefault(country, []).extend(dict(aoi_id=aoi_id, split=s, geometry=geom) for s in chip_splits)
        print(f"{country} {aoi_id}: {shape3[1]} x {shape3[0]} px at {RES} m, offset {offset}, splits {chip_splits}",
              flush=True)

    gpd.GeoDataFrame(index, crs=4326).to_parquet(out / "planet/index.parquet")
    for country, rows in tables.items():
        path = out / "ftw" / country / f"chips_{country}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        gpd.GeoDataFrame(rows, crs=4326).to_parquet(path)
    counts = pd.DataFrame([dict(country=c, split=s) for (c, _), ss in splits.items() for s in ss])
    print(counts.value_counts().unstack(fill_value=0).to_string())
    print(f"-> {out}")


if __name__ == "__main__":
    main()
