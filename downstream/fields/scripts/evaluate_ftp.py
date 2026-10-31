#!/usr/bin/env python3
"""Score a trained checkpoint with the paper's own evaluation (watershed + D4 test-time augmentation, polygon
metrics), on a folder written by scripts/make_ftp_layout.py, and draw prediction panels.

    python scripts/evaluate_ftp.py --ckpt runs/<name>/checkpoints/last.ckpt --root data/ftp_layout/<layout> \
        --countries belgium lithuania latvia croatia cambodia south_africa --out results/<name>

Writes results/<name>/metrics.csv (one row per country, the paper's columns), prints the macro-average the paper
reports (PQ, SQ, RQ, F1[.5:.95], object F1, pixel IoU, polygon-count error, boundary error in metres) and, with
--panels N, saves N panels per country to results/<name>/panels/<country>/<aoi_id>.jpg:
    input (window b) | labels | predicted classes | predicted fields (watershed) with the labelled outlines
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_dilation, distance_transform_edt

from ftw_planet.datasets import PLANET_SR_SCALE, FTWPlanet
from ftw_planet.eval import evaluate_checkpoint
from ftw_planet.evaluation import macro_average
from ftw_planet.inference import load_task, pad_to_min, predict_tta, watershed_instances

ROOT = Path(__file__).resolve().parent.parent
ROWS = [("PQ", "pq"), ("SQ", "pq_sq"), ("RQ", "pq_rq"), ("F1[.5:.95]", "ap_5_95"), ("Obj F1 (WS+TTA)", "object_ws_f1"),
        ("Pixel IoU", "pixel_level_iou"), ("|delta N_poly|", "polygon_count_delta_mean"),
        ("Boundary err (m)", "boundary_error_m_mean")]
CLASS_RGB = np.array([[40, 30, 25], [190, 230, 140], [255, 255, 255], [128, 128, 128]], np.uint8)


def stretch(rgb: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(rgb, [1, 99])
    return (np.clip((rgb - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)


def panels(task, root: Path, countries: list[str], split: str, per_country: int, out: Path, device,
           s2_root: Path | None = None) -> None:
    font = ImageFont.load_default(size=18)
    rng = np.random.default_rng(0)
    for country in countries:
        ds = FTWPlanet(root=str(root), countries=[country], split=split, transforms=None, load_boundaries=True)
        for i in range(min(per_country, len(ds))):
            sample, rec = ds[i], ds.records[i]
            image = sample["image"][None].to(device) / PLANET_SR_SCALE
            mask = sample["mask"][None].to(device)
            image, _, h, w = pad_to_min(image, mask, min_size=512)
            seg = predict_tta(task.model, image).argmax(1)[0, :h, :w].cpu().numpy().astype(np.uint8)
            inst = watershed_instances(seg, distance_transform_edt(seg != 2).astype(np.float32), h_min=2.0)
            gt = sample["mask"].numpy()
            rgb = stretch(sample["image"][:3].numpy().transpose(1, 2, 0))       # window b, R G B
            s2_path = s2_root / country / "s2_images/window_b" / f"{rec['patch_id']}.tif" if s2_root else None
            s2 = None
            if s2_path and s2_path.exists():                                     # the FTW chip (B04 B03 B02)
                import rasterio
                with rasterio.open(s2_path) as f:
                    s2 = stretch(f.read((1, 2, 3)).transpose(1, 2, 0).astype(np.float32))
            colours = rng.integers(60, 255, (inst.max() + 1, 3)).astype(np.uint8)
            colours[0] = 0
            fields = colours[inst]
            fields[binary_dilation(gt == 2) & (gt != 3)] = (255, 235, 0)
            cells = [("SR input, window b", rgb), ("labels", CLASS_RGB[gt]), ("predicted classes", CLASS_RGB[seg]),
                     ("predicted fields + labelled outlines", fields)]
            if s2 is not None:          # enlarged to the panel size for display only (nearest neighbour)
                cells.insert(0, (f"Sentinel-2 (FTW chip, {s2.shape[1]} x {s2.shape[0]} px), window b", s2))
            k = 2 if max(h, w) < 700 else 1
            sheet = Image.new("RGB", (len(cells) * (w * k + 6), h * k + 56), "white")
            draw = ImageDraw.Draw(sheet)
            draw.text((4, 2), f"{country} {rec['patch_id']}   {inst.max()} predicted fields", fill="black", font=font)
            for j, (title, cell) in enumerate(cells):
                draw.text((j * (w * k + 6) + 4, 26), title, fill="black", font=font)
                cell = cell if cell.shape[:2] == (h, w) else np.asarray(Image.fromarray(cell).resize((w, h), Image.NEAREST))
                sheet.paste(Image.fromarray(cell).resize((w * k, h * k), Image.NEAREST), (j * (w * k + 6), 54))
            path = out / "panels" / country / f"{rec['patch_id']}.jpg"
            path.parent.mkdir(parents=True, exist_ok=True)
            sheet.save(path, quality=90)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--countries", nargs="+", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--panels", type=int, default=0, help="panels per country")
    ap.add_argument("--s2-root", type=Path, default=ROOT / "data/ftw",
                    help="FTW chips for the Sentinel-2 column of the panels (left out if a chip is not there)")
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()
    root = args.root if args.root.is_absolute() else ROOT / args.root
    out = args.out if args.out.is_absolute() else ROOT / args.out

    csv = evaluate_checkpoint(args.ckpt, out / "metrics.csv", tuple(args.countries), root=str(root),
                              split=args.split, num_workers=args.num_workers, backend="planet", min_pad_size=512)
    summary = macro_average(csv, tuple(args.countries))
    print(f"\nmacro-average over {summary['n_countries']} of {summary['n_expected']} countries ({args.split}):")
    for label, key in ROWS:
        if key in summary:
            print(f"  {label:18s} {summary[key]:.4f}")
    if args.panels:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        panels(load_task(args.ckpt, device), root, args.countries, args.split, args.panels, out, device, args.s2_root)
        print(f"panels -> {out / 'panels'}")


if __name__ == "__main__":
    main()
