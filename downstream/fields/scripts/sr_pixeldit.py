#!/usr/bin/env python3
"""Super-resolve Fields of The World chips with PixelDiT (RGB model), window by window.

A chip is resampled to its 10 m UTM grid (ftw_geo.read_lr) and covered with overlapping 48 x 48 px windows; each
window and each of the two dates becomes one 1584 x 1584 px image at 0.303 m:

    python scripts/sr_pixeldit.py --select data/selections/first_check.txt

Output: data/sr/pixeldit_rgb/<country>/<aoi_id>/<window_a|window_b>_r<row>_c<col>.npy, uint8 (3, 1584, 1584), where
row / col are the window's offsets on the 10 m grid. Resumable: existing files are skipped. scripts/mosaic.py puts
the windows of a chip together.

Sentinel-2 L2A products of 2022 and later carry a +1000 DN offset and the model was trained on a mix of both, so
chips are used as delivered. `--offset flip` runs the other convention (adds 1000 where the chip has no offset,
removes it where it has one; the chip's convention is the `offset` column of data/chip_stats.csv), to see how much
it matters.

Loading and sampling come from the `superres` package of super-res/super-resolution-seg-cls (installed in envs/env_sr):
EMA weights, 50 steps, conditioning normalised with the training band statistics. ~3 s per window on an H100.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ftw_geo import LR_PATCH, ROOT, WINDOWS, chip_grid, chip_path, chip_windows, read_lr, read_selection
from superres.models.pixeldit import generate_hr, load_denoiser, load_state, make_sampler, prepare_lr_cond, rgb_stats

SEGCLS = ROOT.parents[2] / "super-resolution-seg-cls"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--select", type=Path, required=True, help="text file with `<country> <aoi_id>` per line")
    ap.add_argument("--ftw", type=Path, default=ROOT / "data/ftw")
    ap.add_argument("--out", type=Path, default=None, help="default: data/sr/pixeldit_rgb[_flip]")
    ap.add_argument("--offset", default="keep", choices=("keep", "flip"))
    ap.add_argument("--ckpt", type=Path, default=SEGCLS / "weights/pixeldit_rgb.ckpt")
    ap.add_argument("--stats", type=Path, default=SEGCLS / "configs/s2_band_stats.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--limit", type=int, default=None, help="only the first N windows (for a check)")
    ap.add_argument("--mmap", action="store_true",
                    help="memory-map the checkpoint instead of reading it into RAM (for nodes with little memory)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    out = args.out or ROOT / "data/sr" / ("pixeldit_rgb" if args.offset == "keep" else "pixeldit_rgb_flip")

    shift = {}
    if args.offset == "flip":
        stats = pd.read_csv(ROOT / "data/chip_stats.csv")
        shift = {(r.country, r.aoi_id): 1000.0 - 2 * r.offset for r in stats.itertuples()}

    todo = []
    for country, aoi_id in read_selection(args.select):
        grid = chip_grid(chip_path(args.ftw, country, aoi_id, WINDOWS[0]))
        for window in WINDOWS:
            for row, col in chip_windows(grid):
                path = out / country / aoi_id / f"{window}_r{row}_c{col}.npy"
                if not path.exists():
                    todo.append((country, aoi_id, window, row, col, path))
    todo = todo[: args.limit]
    print(f"{len(todo)} windows to generate -> {out}", flush=True)
    if not todo:
        return

    print(f"loading {args.ckpt} ...", flush=True)
    state = (torch.load(args.ckpt, map_location="cpu", mmap=True, weights_only=False)["state_dict"] if args.mmap
             else load_state(args.ckpt))
    net = load_denoiser(state, "ema_denoiser.", args.device)
    del state
    lo, scale = rgb_stats(args.stats)
    sampler = make_sampler(args.device, num_steps=args.steps)
    print(f"{torch.cuda.get_device_name(0)} | steps: {args.steps} | seed: {args.seed} | offset: {args.offset}",
          flush=True)

    chip, lr = None, None
    started = time.time()
    for done, (country, aoi_id, window, row, col, path) in enumerate(todo, 1):
        if chip != (country, aoi_id, window):
            chip = (country, aoi_id, window)
            source = chip_path(args.ftw, country, aoi_id, window)
            lr, _ = read_lr(source, chip_grid(source))
            lr = lr + shift.get((country, aoi_id), 0.0)
        patch = torch.from_numpy(lr[None, :, row : row + LR_PATCH, col : col + LR_PATCH].copy())
        image = generate_hr(net, sampler, prepare_lr_cond(patch, lo, scale).to(args.device), args.device,
                            seed=args.seed)[0]
        array = ((image + 1) / 2).clamp(0, 1).mul(255).round().to(torch.uint8).cpu().numpy()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part.npy")
        np.save(tmp, array)
        tmp.replace(path)
        if done % 10 == 0 or done == len(todo):
            rate = (time.time() - started) / done
            print(f"  {done}/{len(todo)}  {rate:.1f} s/window  eta {(len(todo) - done) * rate / 60:.0f} min", flush=True)
    print(f"done: {len(todo)} windows in {(time.time() - started) / 60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
