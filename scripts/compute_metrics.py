"""Score every generated model against the ground truth on the same test tiles.

Stage 2 of 2 (stage 1 is scripts/generate_sr.py). Runs in envs/adcsr, which has
torchmetrics + torch-fidelity + transformers; envs/srbench does not.

    envs/adcsr/bin/python scripts/compute_metrics.py --root results/benchmark/test_mini \
        --models ESRGAN EDSR SwinIR

srbench port of the project's compute_metrics.py: the metric code is unchanged; removed are the split-file filter
(the tile sample is chosen by generate_sr.py) and the experiment-numbered text table; --table-csv writes the
benchmark table (default <root>/results/<root name>.csv) as
    ,SSIM,PSNR,LPIPS,MSE,Inception_Score,CLIP_SCORE,FID
with FID and Inception_Score from the native-resolution PATCH protocol (below).

METRICS AND EXACTLY HOW EACH IS COMPUTED
All images are uint8 RGB, 1584x1584 (the x4/x8 outputs upsampled bicubically by
generate_sr.py); per-image metrics use [0, 1] floats.

  MSE    mean squared error over all pixels and channels, per tile, then averaged.
  PSNR   10*log10(1 / MSE) per tile (data range 1.0), then averaged.
  SSIM   torchmetrics SSIM on RGB, Gaussian window 11 / sigma 1.5, data range
         1.0, per tile, then averaged.
  LPIPS  AlexNet LPIPS (the standard variant) at full resolution, per tile.
  FID    Inception-v3 pool features (torch-fidelity weights, the reference FID
         implementation), computed TWO ways because they measure different things:
           tile  -- each whole 1584px tile is resized to 299px. The standard
                    protocol, but the 5.3x shrink erases most of the detail a 33x
                    super-resolution model adds, so it mainly judges layout/colour.
           patch -- each tile is cut into a 5x5 grid of 299px crops at NATIVE
                    resolution (no resizing), 25 per tile. This is the one that
                    sees texture and sharpness, and 25x more samples also makes
                    the estimate far more stable.
  IS     Inception Score, same two protocols, 10 splits.
  CLIPScore
         CLIPScore needs a caption, and these tiles have none. The reference-image
         form is used instead: 100 * max(cos(CLIP(pred), CLIP(ground truth)), 0),
         with CLIP ViT-L/14-336 image embeddings. It asks "does the prediction
         depict the same scene, semantically?" -- the SR-appropriate question.

CAVEATS BAKED INTO THE READING, not just the docs:
  * PSNR, SSIM and MSE reward blur at this upscaling factor (a sharp edge in a
    slightly wrong place scores worse than a smear). The bicubic reference row is
    there to show where pure blur lands on those three.
  * IS and FID use an ImageNet classifier. Satellite tiles are far outside that
    domain, so IS in particular is weakly meaningful here; compare it only
    across rows, never against published numbers.
  * FID is biased by sample count. Every row uses the same count, so rows are
    comparable to each other but not to FIDs computed on other set sizes.

Per-image metrics are reported as mean +/- 95% CI (1.96 * sd / sqrt(n)); two
models whose intervals overlap heavily are not reliably different.
"""
import argparse
import csv
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

PATCH = 299
CLIP_NAME = "openai/clip-vit-large-patch14-336"
GT_DIR = "groundtruth"
BICUBIC_DIR = "bicubic_downsampled_gt"


def load_u8(path):
    return torch.from_numpy(np.array(Image.open(path).convert("RGB"))).permute(2, 0, 1).contiguous()


class Pairs(Dataset):
    def __init__(self, pred_dir, gt_dir, tiles):
        self.pred_dir, self.gt_dir, self.tiles = pred_dir, gt_dir, tiles

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, i):
        t = self.tiles[i]
        pred = load_u8(os.path.join(self.pred_dir, t)) if self.pred_dir else torch.empty(0)
        return t, pred, load_u8(os.path.join(self.gt_dir, t))


def grid_patches(x):
    """(B,3,H,W) -> (B*n*n,3,299,299), a centred n x n grid of native-res crops."""
    _, _, h, w = x.shape
    n = min(h, w) // PATCH
    oy, ox = (h - n * PATCH) // 2, (w - n * PATCH) // 2
    crops = [x[:, :, oy + r * PATCH:oy + (r + 1) * PATCH, ox + c * PATCH:ox + (c + 1) * PATCH]
             for r in range(n) for c in range(n)]
    return torch.cat(crops, dim=0)


class Clip:
    def __init__(self, device):
        from transformers import CLIPImageProcessor, CLIPModel
        self.model = CLIPModel.from_pretrained(CLIP_NAME).to(device).eval()
        proc = CLIPImageProcessor.from_pretrained(CLIP_NAME)
        self.size = proc.crop_size["height"]
        self.mean = torch.tensor(proc.image_mean, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(proc.image_std, device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def embed(self, u8):
        # Tiles are square, so CLIP's resize-shortest-side + centre-crop is just a
        # resize; doing it on the GPU avoids a slow PIL round trip per tile.
        x = F.interpolate(u8.float() / 255.0, size=(self.size, self.size),
                          mode="bicubic", antialias=True, align_corners=False).clamp(0, 1)
        out = self.model.get_image_features(pixel_values=(x - self.mean) / self.std)
        if not torch.is_tensor(out):  # newer transformers return an output object
            emb = getattr(out, "image_embeds", None)
            out = emb if emb is not None else out.pooler_output
        return F.normalize(out.float(), dim=-1)


def seeded_is(metric, seed=0):
    """InceptionScore shuffles features into splits with the global RNG, so the
    same images give a slightly different score on every run (seen: 1.53 vs 1.54
    for byte-identical sets). Seed it so the table is reproducible."""
    torch.manual_seed(seed)
    return [float(v) for v in metric.compute()]


def mean_ci(values):
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return float("nan"), float("nan")
    ci = 1.96 * v.std(ddof=1) / math.sqrt(len(v)) if len(v) > 1 else float("nan")
    return float(v.mean()), float(ci)


def list_pngs(d):
    return sorted(f for f in os.listdir(d) if f.endswith(".png") and not f.endswith(".tmp.png"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--models", nargs="*", default=None,
                    help="model directory names under --root, in table order "
                         "(default: every directory with a meta.json, sorted)")
    ap.add_argument("--out-dir", default=None, help="default: <root>/results")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--is-splits", type=int, default=10)
    ap.add_argument("--device", default="cuda", help="cpu works too, slowly -- useful for testing")
    ap.add_argument("--table-csv", default=None,
                    help="benchmark table (,SSIM,PSNR,LPIPS,MSE,Inception_Score,CLIP_SCORE,FID); "
                         "default: <out-dir>/<root name>.csv")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # the summary printed at the end has ± and arrows

    from torchmetrics.functional.image import structural_similarity_index_measure as ssim_fn
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.inception import InceptionScore
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    device = args.device
    out_dir = args.out_dir or os.path.join(args.root, "results")
    os.makedirs(out_dir, exist_ok=True)
    gt_dir = os.path.join(args.root, GT_DIR)
    tiles = list_pngs(gt_dir)
    if not tiles:
        raise SystemExit(f"no ground-truth PNGs in {gt_dir}; run generate_sr.py --reference first")
    table_csv = args.table_csv or os.path.join(out_dir, os.path.basename(os.path.normpath(args.root)) + ".csv")

    if args.models is None:
        models = sorted(d for d in os.listdir(args.root)
                        if os.path.isfile(os.path.join(args.root, d, "meta.json"))
                        and d not in (GT_DIR, BICUBIC_DIR))
    else:
        models = list(args.models)
    rows_to_score = models + [BICUBIC_DIR]

    # Refuse to score an incomplete model. A partially generated set would be
    # compared on a different subset of tiles -- and FID on a smaller sample is
    # biased -- which silently breaks the "same test set" guarantee.
    problems = []
    for m in rows_to_score:
        d = os.path.join(args.root, m)
        have = set(list_pngs(d)) if os.path.isdir(d) else set()
        missing = [t for t in tiles if t not in have]
        if missing:
            problems.append(f"  {m}: {len(missing)} of {len(tiles)} tiles missing (e.g. {missing[0]})")
    if problems:
        raise SystemExit("incomplete generation -- finish these (resubmit run_generate.sh) before scoring:\n"
                         + "\n".join(problems))
    print(f"scoring {len(rows_to_score)} row(s) on {len(tiles)} tiles", flush=True)

    fid_kw = dict(feature=2048, normalize=False, reset_real_features=False)
    fid_tile = FrechetInceptionDistance(**fid_kw).to(device)
    fid_patch = FrechetInceptionDistance(**fid_kw).to(device)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device)
    clip = Clip(device)
    splits = max(1, min(args.is_splits, len(tiles)))

    def loader(pred_dir):
        return DataLoader(Pairs(pred_dir, gt_dir, tiles), batch_size=1,
                          num_workers=args.workers, pin_memory=True)

    # ---- pass 1: ground truth (FID real statistics, reference IS, CLIP) ----------
    t0 = time.time()
    is_gt_tile = InceptionScore(normalize=False, splits=splits).to(device)
    is_gt_patch = InceptionScore(normalize=False, splits=max(1, min(args.is_splits, 25 * len(tiles)))).to(device)
    gt_clip = {}
    with torch.no_grad():
        for (name,), _, gt in loader(None):
            gt = gt.to(device, non_blocking=True)
            patches = grid_patches(gt)
            fid_tile.update(gt, real=True)
            fid_patch.update(patches, real=True)
            is_gt_tile.update(gt)
            is_gt_patch.update(patches)
            gt_clip[name] = clip.embed(gt)
    gt_is_t = seeded_is(is_gt_tile)
    gt_is_p = seeded_is(is_gt_patch)
    print(f"ground truth pass done in {time.time() - t0:.0f}s", flush=True)

    summary = [{
        "row": GT_DIR, "label": "ground truth (target)", "step": "", "cond_channels": "",
        "is_tile": gt_is_t[0], "is_tile_std": gt_is_t[1],
        "is_patch": gt_is_p[0], "is_patch_std": gt_is_p[1], "clipscore": 100.0,
    }]
    per_image = []

    # ---- pass 2: each model -------------------------------------------------------
    for m in rows_to_score:
        t0 = time.time()
        meta_path = os.path.join(args.root, m, "meta.json")
        meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
        fid_tile.reset()   # keeps the real statistics (reset_real_features=False)
        fid_patch.reset()
        is_tile = InceptionScore(normalize=False, splits=splits).to(device)
        is_patch = InceptionScore(normalize=False, splits=max(1, min(args.is_splits, 25 * len(tiles)))).to(device)
        vals = {k: [] for k in ("mse", "psnr", "ssim", "lpips", "clipscore")}
        with torch.no_grad():
            for (name,), pred, gt in loader(os.path.join(args.root, m)):
                pred = pred.to(device, non_blocking=True)
                gt = gt.to(device, non_blocking=True)
                patches = grid_patches(pred)
                fid_tile.update(pred, real=False)
                fid_patch.update(patches, real=False)
                is_tile.update(pred)
                is_patch.update(patches)

                p01, g01 = pred.float() / 255.0, gt.float() / 255.0
                mse = float(((p01 - g01) ** 2).mean())
                psnr = 10.0 * math.log10(1.0 / mse) if mse > 0 else float("inf")
                ssim = float(ssim_fn(p01, g01, data_range=1.0))
                lp = float(lpips(p01, g01))
                lpips.reset()
                cs = 100.0 * max(float((clip.embed(pred) * gt_clip[name]).sum()), 0.0)
                for k, v in zip(vals, (mse, psnr, ssim, lp, cs)):
                    vals[k].append(v)
                per_image.append({"row": m, "tile": name, "mse": mse, "psnr": psnr,
                                  "ssim": ssim, "lpips": lp, "clipscore": cs})

        is_t = seeded_is(is_tile)
        is_p = seeded_is(is_patch)
        row = {
            "row": m,
            # The weights are part of the identity of a row: the same checkpoint
            # scored from the raw vs EMA weights is two different images sets.
            "label": (meta.get("label") or ("bicubic of GT downsampled to 48px (reference, uses HR)"
                                            if m == BICUBIC_DIR else m))
                     + (" [raw weights]" if m.endswith("__raw") else ""),
            "weights": meta.get("weights", ""),
            "step": meta.get("step", ""), "cond_channels": meta.get("cond_channels", ""),
            "fid_tile": float(fid_tile.compute()), "fid_patch": float(fid_patch.compute()),
            "is_tile": is_t[0], "is_tile_std": is_t[1], "is_patch": is_p[0], "is_patch_std": is_p[1],
        }
        for k, v in vals.items():
            row[k], row[k + "_ci95"] = mean_ci(v)
        summary.append(row)
        print(f"{m}: PSNR {row['psnr']:.2f}  SSIM {row['ssim']:.4f}  LPIPS {row['lpips']:.4f}  "
              f"FID tile {row['fid_tile']:.2f} / patch {row['fid_patch']:.2f}  "
              f"CLIP {row['clipscore']:.2f}  ({time.time() - t0:.0f}s)", flush=True)

    # ---- outputs -----------------------------------------------------------------
    with open(os.path.join(out_dir, "per_image.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_image[0].keys()))
        w.writeheader()
        w.writerows(per_image)
    keys = sorted({k for r in summary for k in r}, key=lambda k: (k not in ("row", "label", "step"), k))
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(summary)
    # benchmark table: FID / IS are the native-resolution PATCH versions
    with open(table_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["", "SSIM", "PSNR", "LPIPS", "MSE", "Inception_Score", "CLIP_SCORE", "FID"])
        for r in summary:
            if r["row"] in (GT_DIR, BICUBIC_DIR):
                continue
            w.writerow([r["label"], f"{r['ssim']:.4f}", f"{r['psnr']:.2f}", f"{r['lpips']:.4f}",
                        f"{r['mse']:.5f}", f"{r['is_patch']:.2f}", f"{r['clipscore']:.2f}",
                        f"{r['fid_patch']:.2f}"])
    print(f"table -> {table_csv}")
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({"num_tiles": len(tiles), "rows": summary}, f, indent=2)

    def fmt(r, k, nd, ci=True):
        if k not in r or r[k] == "" or (isinstance(r[k], float) and not math.isfinite(r[k])):
            return "—"
        s = f"{r[k]:.{nd}f}"
        if ci and (k + "_ci95") in r and math.isfinite(r[k + "_ci95"]):
            s += f" ± {r[k + '_ci95']:.{nd}f}"
        return s

    lines = [
        f"# Benchmark on {len(tiles)} held-out test tiles",
        "",
        "Arrows show which direction is better. ± is a 95% confidence interval over tiles.",
        "",
        "| model | step | bands | PSNR ↑ | SSIM ↑ | LPIPS ↓ | MSE ↓ | FID tile ↓ | FID patch ↓ "
        "| IS tile ↑ | IS patch ↑ | CLIPScore ↑ |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in summary:
        lines.append(
            f"| {r['label']} | {r.get('step', '')} | {r.get('cond_channels', '')} "
            f"| {fmt(r, 'psnr', 2)} | {fmt(r, 'ssim', 4)} | {fmt(r, 'lpips', 4)} | {fmt(r, 'mse', 5)} "
            f"| {fmt(r, 'fid_tile', 2, False)} | {fmt(r, 'fid_patch', 2, False)} "
            f"| {fmt(r, 'is_tile', 2, False)} | {fmt(r, 'is_patch', 2, False)} | {fmt(r, 'clipscore', 2)} |")
    lines += ["",
              "- PSNR/SSIM/MSE reward blur at this scale; the bicubic reference row shows where pure blur lands.",
              "- FID patch (native-resolution 299px crops) is the FID that can see sharpness; "
              "FID tile shrinks each tile 5.3x first.",
              "- IS uses an ImageNet classifier on satellite imagery: compare rows only.",
              "- CLIPScore here is prediction-vs-ground-truth image similarity (no captions exist)."]
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as f:  # ± and arrows
        f.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\nwrote {table_csv}, {out_dir}/summary.md, summary.csv, summary.json, per_image.csv")


if __name__ == "__main__":
    main()
