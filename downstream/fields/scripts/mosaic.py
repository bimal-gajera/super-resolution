#!/usr/bin/env python3
"""Put the PixelDiT windows of each chip and date together into one image on the chip's UTM grid at 1 m.

    python scripts/mosaic.py --select data/selections/first_check.txt

Per window: the 0.303 m output is registered against its own 10 m input (the model's output is shifted by ~7 m,
inherited from its training pairs; the repo's scripts/registration.py measures the shift to 1.25 m), moved
back, box-averaged to 1 m and blended into the mosaic with weights that fall off towards the window edge, so that
tone steps between windows are spread over the overlap. A window whose own estimate is unreliable takes the median
shift of the chip's reliable windows.

Output: data/mosaic/<run>/<country>/<aoi_id>_<window>.npy, uint8 (H, W, 3) at 1 m (georeferencing:
ftw_geo.chip_grid(...).transform(1.0)); pixels no window covers are 0 in all bands. data/mosaic/<run>/offsets.csv
has one row per window (dx, dy in Sentinel-2 pixels, x = east, y = south, and the correlations).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ftw_geo import (LR_PATCH, ROOT, S2_M, WINDOWS, area_resize, chip_grid, chip_path, chip_windows, read_lr,
                     read_selection)

sys.path.insert(0, str(ROOT.parents[1] / "scripts"))
from registration import estimate_offset  # noqa: E402

M = int(LR_PATCH * S2_M)        # window size in the 1 m mosaic
FALLBACK = (-0.62, -0.35)       # median shift measured on xBD-S12, used if no window of a chip is reliable


def feather(size: int = M, ramp: int = 60) -> np.ndarray:
    edge = np.minimum(np.arange(size) + 0.5, size - np.arange(size) - 0.5)
    w = np.clip(edge / ramp, 1e-3, 1.0)
    return np.outer(w, w)


def reliable(dx: float, dy: float, corr: float) -> bool:
    return corr > 0.6 and abs(dx) < 2 and abs(dy) < 2


def build(windows: list[dict], grid_shape: tuple[int, int]) -> np.ndarray:
    h, w = grid_shape
    acc, weight, wt = np.zeros((h, w, 3)), np.zeros((h, w)), feather()
    for win in windows:
        r0 = int(round((win["row"] - win["use_dy"]) * S2_M))
        c0 = int(round((win["col"] - win["use_dx"]) * S2_M))
        rs, cs = slice(max(r0, 0), min(r0 + M, h)), slice(max(c0, 0), min(c0 + M, w))
        src = (slice(rs.start - r0, rs.stop - r0), slice(cs.start - c0, cs.stop - c0))
        acc[rs, cs] += win["image"][src] * wt[src][..., None]
        weight[rs, cs] += wt[src]
    out = acc / np.maximum(weight, 1e-9)[..., None]
    out[weight == 0] = 0
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--select", type=Path, required=True)
    ap.add_argument("--ftw", type=Path, default=ROOT / "data/ftw")
    ap.add_argument("--sr", type=Path, default=ROOT / "data/sr/pixeldit_rgb")
    ap.add_argument("--out", type=Path, default=None, help="default: data/mosaic/<name of --sr>")
    args = ap.parse_args()
    out = args.out or ROOT / "data/mosaic" / args.sr.name

    rows = []
    for country, aoi_id in read_selection(args.select):
        for window in WINDOWS:
            source = chip_path(args.ftw, country, aoi_id, window)
            grid = chip_grid(source)
            paths = {(r, c): args.sr / country / aoi_id / f"{window}_r{r}_c{c}.npy" for r, c in chip_windows(grid)}
            if not all(p.exists() for p in paths.values()):
                print(f"{country} {aoi_id} {window}: SR windows missing, skipped")
                continue
            lr, _ = read_lr(source, grid)
            gray = lr.mean(0).astype(np.float64)
            wins = []
            for (r, c), path in paths.items():
                image = np.load(path).transpose(1, 2, 0)
                dx, dy, corr, corr0 = estimate_offset(area_resize(image.mean(2), 8 * LR_PATCH),
                                                      gray[r : r + LR_PATCH, c : c + LR_PATCH])
                wins.append(dict(row=r, col=c, dx=dx, dy=dy, corr=corr, corr_at_zero=corr0,
                                 reliable=reliable(dx, dy, corr), image=area_resize(image, M)))
            good = [w for w in wins if w["reliable"]]
            med = (np.median([w["dx"] for w in good]), np.median([w["dy"] for w in good])) if good else FALLBACK
            for w in wins:
                w["use_dx"], w["use_dy"] = (w["dx"], w["dy"]) if w["reliable"] else med
            path = out / country / f"{aoi_id}_{window}.npy"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, build(wins, grid.shape(1.0)))
            rows += [dict(country=country, aoi_id=aoi_id, window=window,
                          **{k: v for k, v in w.items() if k != "image"}) for w in wins]
            print(f"{country} {aoi_id} {window}: {len(wins)} windows, {len(good)} reliable, shift median "
                  f"({med[0]:+.2f}, {med[1]:+.2f}) S2 px", flush=True)

    if not rows:
        return
    off = pd.DataFrame(rows)
    table = out / "offsets.csv"
    if table.exists():      # keep the rows of chips mosaicked in earlier calls
        old = pd.read_csv(table)
        keys = set(zip(off.country, off.aoi_id, off.window))
        off = pd.concat([old[[k not in keys for k in zip(old.country, old.aoi_id, old.window)]], off])
    off.to_csv(table, index=False)
    good = off[off.reliable]
    print(f"{len(off)} windows, {len(good)} reliable | dx median {good.dx.median():+.2f} "
          f"(25-75 % {good.dx.quantile(.25):+.2f}..{good.dx.quantile(.75):+.2f}) | dy median {good.dy.median():+.2f} "
          f"({good.dy.quantile(.25):+.2f}..{good.dy.quantile(.75):+.2f}) S2 px | correlation with the input "
          f"{good['corr'].median():.2f} (at zero shift {good.corr_at_zero.median():.2f}) -> {out}")


if __name__ == "__main__":
    main()
