#!/usr/bin/env python3
"""Where does each flow put the edge of a labelled field outline? (Placement only, not clutter.)

scripts/first_check.py marks each image's strongest edges anywhere in the chip, so sharp houses, roads and tree
crowns count against a flow. This script looks only across the outlines: on the 1 m grid, the pixels closer to one
outline pixel than to any other (up to REACH metres on both sides) form a strip across the outline; the distance of
the strip's strongest-gradient pixel from the outline is the flow's placement error at that point.

    python scripts/outline_offset.py --select data/selections/first_check.txt

Writes results/<out>/outline_offset.csv (one row per chip, date and flow): median error and the share of outline
points placed within 3 and 5 m, for all points and for the "visible" half (points where the edge is strong, ranked
by the mean of the flows' own edge strengths, so the subset is the same for every flow). `chance_3m` is what a
random position in the strip scores.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import ndimage

from ftw_geo import (ROOT, WINDOWS, chip_grid, chip_path, footprint, label_rasters, load_polygons, read_lr,
                     read_selection)

REACH = 15      # metres on either side of the outline
MARGIN = 25     # metres dropped along the chip edge
SIGMA = 1.5     # metres, smoothing of the gradient (about the 3 m pixel the downstream model sees)


def strips(line: np.ndarray, valid: np.ndarray):
    dist, (iy, ix) = ndimage.distance_transform_edt(~line, return_indices=True)
    mask = valid & (dist <= REACH)
    key = (iy * line.shape[1] + ix)[mask]
    order = np.argsort(key, kind="stable")
    key = key[order]
    start = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    size = np.diff(np.r_[start, len(key)])
    return mask, order, start, size, dist[mask][order]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--select", type=Path, required=True)
    ap.add_argument("--ftw", type=Path, default=ROOT / "data/ftw")
    ap.add_argument("--polygons", type=Path, default=ROOT / "data/ftw_polygons")
    ap.add_argument("--sr", nargs="+", default=["our SR=data/mosaic/pixeldit_rgb"], metavar="NAME=FOLDER")
    ap.add_argument("--out", type=Path, default=ROOT / "results/first_check")
    args = ap.parse_args()
    sr_flows = {name: ROOT / folder for name, folder in (item.split("=", 1) for item in args.sr)}

    rows = []
    for country, aoi_id in read_selection(args.select):
        first = chip_path(args.ftw, country, aoi_id, WINDOWS[0])
        grid = chip_grid(first)
        line, _, _ = label_rasters(load_polygons(args.polygons, country, grid), grid, 1.0)
        fp = footprint(first, grid, 1.0)
        valid = ndimage.distance_transform_edt(np.pad(fp, 1))[1:-1, 1:-1] > MARGIN
        if not (line & valid).any():
            continue
        mask, order, start, size, dist = strips(line, valid)
        keep = size >= 2 * REACH                 # strips that reach to both sides
        for window in WINDOWS:
            mosaics = {name: folder / country / f"{aoi_id}_{window}.npy" for name, folder in sr_flows.items()}
            if not all(p.exists() for p in mosaics.values()):
                continue
            lr, _ = read_lr(chip_path(args.ftw, country, aoi_id, window), grid)
            bic = torch.nn.functional.interpolate(torch.from_numpy(lr[None]), scale_factor=10, mode="bicubic",
                                                  align_corners=False)[0].mean(0).numpy()
            images = {"bicubic": bic, **{n: np.load(p).astype(np.float32).mean(2) for n, p in mosaics.items()}}
            err, strength = {}, {}
            for flow, gray in images.items():
                mag = ndimage.gaussian_gradient_magnitude(gray, SIGMA)
                m = mag[mask][order]
                best = np.array([s + np.argmax(m[s : s + n]) for s, n in zip(start, size)])
                err[flow] = dist[best][keep]
                strength[flow] = (m[best] / np.median(mag[valid]))[keep]
            rank = np.mean([pd.Series(s).rank(pct=True).to_numpy() for s in strength.values()], axis=0)
            visible = rank >= np.median(rank)
            for flow, e in err.items():
                rows.append(dict(country=country, aoi_id=aoi_id, window=window, flow=flow, n_points=len(e),
                                 chance_3m=float((dist <= 3).mean()),
                                 error_m=float(np.median(e)), within_3m=float((e <= 3).mean()),
                                 within_5m=float((e <= 5).mean()),
                                 error_m_visible=float(np.median(e[visible])),
                                 within_3m_visible=float((e[visible] <= 3).mean()),
                                 within_5m_visible=float((e[visible] <= 5).mean())))
        print(f"{country} {aoi_id}: done", flush=True)

    scores = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    scores.to_csv(args.out / "outline_offset.csv", index=False)
    cols = ["error_m", "within_3m", "within_5m", "error_m_visible", "within_3m_visible", "within_5m_visible"]
    print("\nmean over chips and dates, per country and flow")
    print(scores.groupby(["country", "flow"])[cols].mean().round(3).to_string())
    print("\nall chips")
    print(scores.groupby("flow")[cols].mean().round(3).to_string())
    print(f"chance (random position in the strip) within 3 m: {scores.chance_3m.mean():.3f}")
    per_chip = scores.groupby(["country", "aoi_id", "flow"])[cols].mean().unstack("flow")
    rng = np.random.default_rng(0)
    print("\ndifference to bicubic (mean over chips, 95 % bootstrap interval over chips)")
    for flow in sr_flows:
        for c in cols:
            d = (per_chip[c][flow] - per_chip[c]["bicubic"]).dropna().to_numpy()
            boot = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            better = int((d < 0).sum() if c.startswith("error") else (d > 0).sum())
            print(f"  {flow:<16} {c:<18} {d.mean():+.3f}  ({lo:+.3f} .. {hi:+.3f})  better in {better} of {len(d)} chips")
    print(f"\n-> {args.out / 'outline_offset.csv'}")


if __name__ == "__main__":
    main()
