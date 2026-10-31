#!/usr/bin/env python3
"""First check on FTW chips: do the edges of our SR sit on the labelled field boundaries, and more tightly than
the edges of bicubic-upsampled Sentinel-2?

    python scripts/first_check.py --select data/selections/first_check.txt

Flows, per chip and date, on the chip's UTM grid at 3 m (the resolution of the paper's PlanetScope images; the SR
mosaic at 1 m averaged over 3 x 3 pixels):
    bicubic   the 10 m input upsampled bicubically
    <name>    each SR mosaic given as --sr name=folder (default: "our SR"=data/mosaic/pixeldit_rgb)

Scores use only the FTW polygons (results/<out>/scores.csv, one row per chip, date and flow):
  - The gradient of the grey image is thinned to ridges (non-maximum suppression) and the K strongest ridge pixels
    are kept, K = the number of pixels on the labelled outlines, so every flow marks the same number of pixels.
  - precision_<t>m: share of the marked pixels within t metres of a labelled outline (t = 3, 6, 10);
    recall_<t>m: share of outline pixels with a marked pixel within t metres;
    chance_<t>m: share of all pixels within t metres of an outline (what a random edge map scores).
  - edge_to_label_m, label_to_edge_m: median distances in true metres.
profile.csv: mean gradient (not thinned, relative to the image's mean) against distance to the nearest outline.
panels/<country>/<aoi_id>_<window>.jpg: top row the whole chip (Sentinel-2 | bicubic | SR | SR with the outlines),
bottom row a zoom on the part with most outlines (Sentinel-2 | bicubic at 3 m | SR at 3 m | SR at 1 m + outlines).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from ftw_geo import (ROOT, WINDOWS, block_mean, chip_grid, chip_path, footprint, label_rasters, load_polygons,
                     read_lr, read_selection)

RES = 3                       # metres, the grid the flows are scored on
TOLERANCES = (3, 6, 10)       # metres
BINS = np.arange(0, 48, 3)    # metres, distance-to-outline bins of the gradient profile
MARGIN = 25                   # metres dropped along the chip edge (not every window reaches it after its shift)
YELLOW = np.array([255, 235, 0], np.uint8)


def thin_edges(gray: np.ndarray, sigma: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Gradient magnitude, and the same with everything but the ridges set to 0 (non-maximum suppression)."""
    gx = ndimage.gaussian_filter(gray, sigma, order=(0, 1))
    gy = ndimage.gaussian_filter(gray, sigma, order=(1, 0))
    mag = np.hypot(gx, gy)
    sector = ((np.rad2deg(np.arctan2(gy, gx)) % 180 + 22.5) // 45).astype(int) % 4
    pad = np.pad(mag, 1, mode="edge")
    h, w = mag.shape
    keep = np.zeros(mag.shape, bool)
    for k, (dy, dx) in enumerate(((0, 1), (1, 1), (1, 0), (1, -1))):       # neighbours along the gradient
        ahead = pad[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]
        behind = pad[1 - dy : 1 - dy + h, 1 - dx : 1 - dx + w]
        keep |= (sector == k) & (mag >= ahead) & (mag >= behind)
    return mag, np.where(keep, mag, 0.0)


def edge_scores(gray: np.ndarray, dist: np.ndarray, valid: np.ndarray) -> tuple[dict, np.ndarray] | None:
    line = (dist <= RES / 2) & valid
    k = int(line.sum())
    if k < 50:
        return None
    mag, thin = thin_edges(gray)
    thin = np.where(valid, thin, 0.0)
    edge = (thin >= np.partition(thin.ravel(), -k)[-k]) & (thin > 0)
    to_edge = ndimage.distance_transform_edt(~edge) * RES
    row = dict(n_outline=k, edge_to_label_m=float(np.median(dist[edge])),
               label_to_edge_m=float(np.median(to_edge[line])))
    for t in TOLERANCES:
        row[f"precision_{t}m"] = float((dist[edge] <= t).mean())
        row[f"recall_{t}m"] = float((to_edge[line] <= t).mean())
        row[f"chance_{t}m"] = float((dist[valid] <= t).mean())
    rel = mag / mag[valid].mean()
    which = np.digitize(dist, BINS) - 1
    profile = np.array([rel[valid & (which == b)].mean() if (valid & (which == b)).any() else np.nan
                        for b in range(len(BINS) - 1)])
    return row, profile


def stretch(lr: np.ndarray, inside: np.ndarray):
    """One common 1-99 % stretch for the three bands (keeps the colour balance); returns the mapping to uint8."""
    lo, hi = np.percentile(lr[:, inside], [1, 99])
    return lambda x: (np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)


def upsample(image: np.ndarray, k: int, shape: tuple[int, int] | None = None) -> np.ndarray:
    """Nearest-neighbour enlargement by k; padded with white to `shape` if given."""
    out = np.repeat(np.repeat(image, k, 0), k, 1)
    if shape:
        full = np.full((*shape, *out.shape[2:]), 255, out.dtype)
        full[: out.shape[0], : out.shape[1]] = out[: shape[0], : shape[1]]
        out = full
    return out


def with_outlines(image: np.ndarray, line: np.ndarray) -> np.ndarray:
    out = image.copy()
    out[line] = YELLOW
    return out


def zoom_box(line: np.ndarray, valid: np.ndarray, zh: int, zw: int) -> tuple[int, int]:
    """Top-left corner of the zh x zw box with most outline pixels that lies in the valid area."""
    density = ndimage.uniform_filter(line.astype(np.float32), (zh, zw), mode="constant")
    inside = ndimage.uniform_filter(valid.astype(np.float32), (zh, zw), mode="constant") > 0.98
    density = np.where(inside, density, -1) if inside.any() else density
    cy, cx = np.unravel_index(np.argmax(density), density.shape)
    return (int(np.clip(cy - zh // 2, 0, line.shape[0] - zh)), int(np.clip(cx - zw // 2, 0, line.shape[1] - zw)))


def panel(path: Path, header: str, top: list[tuple[str, np.ndarray]], bottom: list[tuple[str, np.ndarray]],
          box: tuple[int, int, int, int]) -> None:
    font = ImageFont.load_default(size=20)
    cells = [Image.fromarray(a).resize((a.shape[1] // 2, a.shape[0] // 2), Image.BOX) for _, a in top]
    zooms = [Image.fromarray(upsample(a, 2)) for _, a in bottom]
    cw, ch = cells[0].size
    zw, zh = zooms[0].size
    col = max(cw, zw) + 8
    head, label = 34, 28
    sheet = Image.new("RGB", (col * len(cells) - 8, head + 2 * label + ch + zh + 8), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((6, 4), header, fill="black", font=font)
    for i, ((title, _), cell) in enumerate(zip(top, cells)):
        draw.text((i * col + 6, head), title, fill="black", font=font)
        sheet.paste(cell, (i * col, head + label))
        y0, x0, zh1, zw1 = box
        draw.rectangle((i * col + x0 // 2, head + label + y0 // 2, i * col + (x0 + zw1) // 2,
                        head + label + (y0 + zh1) // 2), outline=(255, 0, 0), width=2)
    for i, ((title, _), zoom) in enumerate(zip(bottom, zooms)):
        y = head + label + ch + 8
        draw.text((i * col + 6, y), title, fill="black", font=font)
        sheet.paste(zoom, (i * col, y + label))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=88)


def summarise(scores: pd.DataFrame, flows: list[str]) -> None:
    cols = ["precision_3m", "precision_6m", "precision_10m", "recall_6m", "edge_to_label_m", "label_to_edge_m"]
    chance = scores.groupby("country")[["chance_3m", "chance_6m", "chance_10m"]].mean()
    print("\nmean over chips and dates, per country and flow")
    print(scores.groupby(["country", "flow"])[cols].mean().round(3).to_string())
    print("\nall chips")
    print(scores.groupby("flow")[cols].mean().round(3).to_string())
    print(f"chance (share of pixels within 3 / 6 / 10 m of an outline): "
          f"{scores.chance_3m.mean():.3f} / {scores.chance_6m.mean():.3f} / {scores.chance_10m.mean():.3f}")
    print(chance.round(3).to_string())
    # paired difference to bicubic, chips resampled (the two dates of a chip are averaged first)
    per_chip = scores.groupby(["country", "aoi_id", "flow"])[cols].mean().unstack("flow")
    rng = np.random.default_rng(0)
    print("\ndifference to bicubic (mean over chips, 95 % bootstrap interval over chips)")
    for flow in flows:
        for c in cols:
            d = (per_chip[c][flow] - per_chip[c]["bicubic"]).dropna().to_numpy()
            if len(d) < 2:
                continue
            boot = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            print(f"  {flow:<16} {c:<16} {d.mean():+.3f}  ({lo:+.3f} .. {hi:+.3f})  better in "
                  f"{int((d < 0).sum() if c.endswith('_m') else (d > 0).sum())} of {len(d)} chips")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--select", type=Path, required=True)
    ap.add_argument("--ftw", type=Path, default=ROOT / "data/ftw")
    ap.add_argument("--polygons", type=Path, default=ROOT / "data/ftw_polygons")
    ap.add_argument("--sr", nargs="+", default=["our SR=data/mosaic/pixeldit_rgb"], metavar="NAME=FOLDER")
    ap.add_argument("--out", type=Path, default=ROOT / "results/first_check")
    ap.add_argument("--no-panels", action="store_true")
    args = ap.parse_args()
    sr_flows = {name: (ROOT / folder if not Path(folder).is_absolute() else Path(folder))
                for name, folder in (item.split("=", 1) for item in args.sr)}

    rows, profiles = [], []
    for country, aoi_id in read_selection(args.select):
        grid = chip_grid(chip_path(args.ftw, country, aoi_id, WINDOWS[0]))
        polys = load_polygons(args.polygons, country, grid)
        line1, _, dist1 = label_rasters(polys, grid, 1.0)
        fp = footprint(chip_path(args.ftw, country, aoi_id, WINDOWS[0]), grid, 1.0)
        valid1 = ndimage.distance_transform_edt(np.pad(fp, 1))[1:-1, 1:-1] > MARGIN
        h3, w3 = fp.shape[0] // RES, fp.shape[1] // RES
        dist3 = dist1[RES // 2 :: RES, RES // 2 :: RES][:h3, :w3]
        valid3 = block_mean(valid1.astype(np.float32), RES) == 1
        for window in WINDOWS:
            mosaics = {name: folder / country / f"{aoi_id}_{window}.npy" for name, folder in sr_flows.items()}
            if not all(p.exists() for p in mosaics.values()):
                print(f"{country} {aoi_id} {window}: mosaic missing, skipped")
                continue
            lr, inside = read_lr(chip_path(args.ftw, country, aoi_id, window), grid)
            bic1 = torch.nn.functional.interpolate(torch.from_numpy(lr[None]), scale_factor=10, mode="bicubic",
                                                   align_corners=False)[0].permute(1, 2, 0).numpy()
            images1 = {"bicubic": bic1, **{name: np.load(p).astype(np.float32) for name, p in mosaics.items()}}
            images3 = {name: block_mean(im, RES) for name, im in images1.items()}
            chip_rows = {}
            for flow, im in images3.items():
                scored = edge_scores(im.mean(2), dist3, valid3)
                if scored is None:
                    continue
                row, profile = scored
                chip_rows[flow] = row
                rows.append(dict(country=country, aoi_id=aoi_id, window=window, flow=flow, n_fields=len(polys), **row))
                profiles += [dict(country=country, aoi_id=aoi_id, window=window, flow=flow, distance_m=int(b),
                                  gradient=float(v)) for b, v in zip(BINS[:-1], profile)]
            if args.no_panels:
                continue
            to8 = stretch(lr, inside)
            s2_1 = upsample(to8(lr.transpose(1, 2, 0)), 10)
            show1 = {"bicubic": to8(bic1), **{n: images1[n].astype(np.uint8) for n in sr_flows}}
            show3 = {"bicubic": to8(images3["bicubic"]),
                     **{n: np.round(images3[n]).astype(np.uint8) for n in sr_flows}}
            first = next(iter(sr_flows))
            thick = ndimage.binary_dilation(line1, iterations=1)
            zh, zw = fp.shape[0] // 4, fp.shape[1] // 4
            y0, x0 = zoom_box(line1, valid1, zh, zw)
            crop = (slice(y0, y0 + zh), slice(x0, x0 + zw))
            top = [("Sentinel-2 input, 10 m", s2_1), ("bicubic upsampling", show1["bicubic"]),
                   *[(f"{n}, 1 m", show1[n]) for n in sr_flows],
                   (f"{first} + FTW field outlines", with_outlines(show1[first], ndimage.binary_dilation(thick)))]
            bottom = [("zoom: Sentinel-2, 10 m", s2_1[crop]),
                      *[(f"{n} at 3 m", upsample(show3[n], RES, fp.shape)[crop]) for n in show3],
                      (f"{first}, 1 m + outlines", with_outlines(show1[first], thick)[crop])]
            text = "   ".join(f"{f}: {r['precision_6m']:.2f}" for f, r in chip_rows.items())
            chance = next(iter(chip_rows.values()))["chance_6m"] if chip_rows else float("nan")
            header = (f"{country} {aoi_id} {window}   {len(polys)} fields   chip {fp.shape[1]} x {fp.shape[0]} m   "
                      f"strongest edges within 6 m of an outline: {text}   (chance {chance:.2f})")
            panel(args.out / "panels" / country / f"{aoi_id}_{window}.jpg", header, top, bottom, (y0, x0, zh, zw))
        print(f"{country} {aoi_id}: done", flush=True)

    if not rows:
        print("nothing scored")
        return
    args.out.mkdir(parents=True, exist_ok=True)
    scores = pd.DataFrame(rows)
    scores.to_csv(args.out / "scores.csv", index=False)
    profile = pd.DataFrame(profiles)
    profile.to_csv(args.out / "profile.csv", index=False)
    summarise(scores, list(sr_flows))
    print("\nmean gradient relative to the image mean, by distance to the nearest outline (m)")
    print(profile.groupby(["distance_m", "flow"]).gradient.mean().unstack("flow").round(2).to_string())
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
