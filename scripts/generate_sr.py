"""Benchmark stage 1 of 2: write super-resolved test tiles as native-resolution PNGs (stage 2: compute_metrics.py).

Every row is scored at the native Maxar resolution (1584 px = 48 x 33, 0.303 m): a model's x4 / x8 / x16 / x32 output
(192 / 384 / 768 / 1536 px) is upsampled bicubically to 1584 px, and the ground truth is the original Maxar tile (not
the downsampled training target).

    # once per benchmark root: the tile sample, ground truth and the bicubic reference row
    python scripts/generate_sr.py --root results/benchmark/test_mini --reference
    # one row per model; its test YAML gives the network, weights and input settings (--force_yml overrides them)
    python scripts/generate_sr.py --root results/benchmark/test_mini --row ESRGAN -opt options/test/ESRGAN/test_ESRGAN_x4.yml

Layout written (what compute_metrics.py reads):
    <root>/tiles.txt                      the tile keys (test-mini: --num-tiles evenly spaced keys of --meta-info)
    <root>/groundtruth/<key>.png          original Maxar tile, 1584 px
    <root>/bicubic_downsampled_gt/<key>.png   GT -> 48 px (antialiased bicubic) -> 1584 px (reference row, uses the HR)
    <root>/<row>/<key>.png + meta.json    model output upsampled to 1584 px
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from os import path as osp

import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, ROOT)

from srbench.data import build_dataloader, build_dataset  # noqa: E402
from srbench.data.s2maxar_dataset import S2MaxarDataset  # noqa: E402
from srbench.models import build_model  # noqa: E402
from srbench.utils.options import parse_options  # noqa: E402

GT_DIR = 'groundtruth'
BICUBIC_DIR = 'bicubic_downsampled_gt'
NATIVE = 1584  # 48 S2 pixels x 33
LR_SIZE = 48
MAXAR_DIR = '/ocean/projects/cis250179p/purohit/Bimal/visual_small/Maxar'


def resize(x, size, antialias=False):
    """(B, 3, h, w) float in [0, 1] -> (B, 3, size, size), bicubic."""
    return F.interpolate(x, size=(size, size), mode='bicubic', align_corners=False, antialias=antialias).clamp(0, 1)


def to_u8(x):
    return (x * 255.0).round().byte().permute(0, 2, 3, 1).cpu().numpy()


def save_pngs(pool, out_dir, keys, imgs):
    return [pool.submit(Image.fromarray(img).save, osp.join(out_dir, f'{key}.png'), compress_level=1)
            for key, img in zip(keys, imgs)]


def write_reference(args, pool):
    """Tile sample, ground truth and the bicubic reference row."""
    keys = S2MaxarDataset({
        'dataroot': osp.join(ROOT, 'datasets/s2maxar'),
        'meta_info_file': osp.join(ROOT, args.meta_info) if not osp.isabs(args.meta_info) else args.meta_info,
        'io_backend': {'type': 'npy'},
        'scale': 4,
        'phase': 'test',
        'max_samples': args.num_tiles,
    }).keys
    for d in (GT_DIR, BICUBIC_DIR):
        os.makedirs(osp.join(args.root, d), exist_ok=True)
    jobs = []
    for i, key in enumerate(keys):
        gt = tifffile.imread(osp.join(args.maxar_dir, f'{key}.tif'))
        assert gt.shape == (NATIVE, NATIVE, 3) and gt.dtype == np.uint8, (key, gt.shape, gt.dtype)
        jobs += save_pngs(pool, osp.join(args.root, GT_DIR), [key], [gt])
        x = torch.from_numpy(gt).permute(2, 0, 1)[None].float().to(args.device) / 255.0
        bic = resize(resize(x, LR_SIZE, antialias=True), NATIVE)
        jobs += save_pngs(pool, osp.join(args.root, BICUBIC_DIR), [key], to_u8(bic))
        if (i + 1) % 100 == 0:
            print(f'reference: {i + 1}/{len(keys)}', flush=True)
    for j in jobs:
        j.result()
    with open(osp.join(args.root, BICUBIC_DIR, 'meta.json'), 'w') as f:
        json.dump({'label': 'bicubic of GT downsampled to 48px (reference, uses HR)'}, f, indent=2)
    with open(osp.join(args.root, 'tiles.txt'), 'w') as f:  # written last: its presence means the set is complete
        f.write('\n'.join(keys) + '\n')
    print(f'reference rows for {len(keys)} tiles -> {args.root}')


def write_row(args, pool):
    """Run one model (its test YAML) on the tiles of <root>/tiles.txt and save 1584 px PNGs."""
    tiles = osp.join(args.root, 'tiles.txt')
    assert osp.exists(tiles), f'{tiles} missing: run with --reference first'
    argv = ['-opt', args.opt] + (['--force_yml'] + args.force_yml if args.force_yml else [])
    opt, _ = parse_options(ROOT, is_train=False, argv=argv)
    dataset_opt = deepcopy(next(v for k, v in sorted(opt['datasets'].items())))
    dataset_opt.update(name=f'benchmark_{args.row}', meta_info_file=tiles)
    if args.batch_size:
        dataset_opt['batch_size_per_gpu'] = args.batch_size
    dataset_opt.pop('max_samples', None)
    dataset = build_dataset(dataset_opt)
    loader = build_dataloader(dataset, dataset_opt, num_gpu=opt['num_gpu'], dist=False, sampler=None, seed=0)
    model = build_model(opt)

    out_dir = osp.join(args.root, args.row)
    os.makedirs(out_dir, exist_ok=True)
    jobs, t0 = [], time.time()
    for data in loader:
        model.feed_data(data)
        model.test()
        sr = model.output.detach().float().clamp(0, 1)
        if sr.shape[-1] != NATIVE:
            sr = resize(sr, NATIVE)
        jobs += save_pngs(pool, out_dir, data['key'], to_u8(sr))
    for j in jobs:
        j.result()
    meta = {
        'label': args.label or args.row,
        'opt': args.opt,
        'force_yml': args.force_yml or [],
        'weights': opt['path'].get('pretrain_network_g'),
        'param_key': opt['path'].get('param_key_g', 'params'),
        'scale': opt['scale'],
        'output': f"{LR_SIZE * opt['scale']} px" + ('' if LR_SIZE * opt['scale'] == NATIVE else f', upsampled (bicubic) to {NATIVE} px'),
        'num_tiles': len(dataset),
    }
    with open(osp.join(out_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'{args.row}: {len(dataset)} tiles in {time.time() - t0:.0f}s -> {out_dir}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', required=True, help='benchmark folder, e.g. results/benchmark/test_mini')
    parser.add_argument('--reference', action='store_true', help='write tiles.txt, ground truth and bicubic rows')
    parser.add_argument('--meta-info', default='datasets/s2maxar/meta_info/test_clean.txt')
    parser.add_argument('--num-tiles', type=int, default=500, help='evenly spaced sample of --meta-info (test-mini)')
    parser.add_argument('--maxar-dir', default=MAXAR_DIR)
    parser.add_argument('--row', help='row (folder) name, e.g. ESRGAN')
    parser.add_argument('--label', help='label in the results table (default: --row)')
    parser.add_argument('-opt', help='test YAML of the model (options/test/...)')
    parser.add_argument('--force_yml', nargs='+', default=None, help='overrides, e.g. path:pretrain_network_g=...')
    parser.add_argument('--batch-size', type=int, default=None, help='default: the test YAML (2 at x16 / x32)')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    args.root = args.root if osp.isabs(args.root) else osp.join(ROOT, args.root)
    os.makedirs(args.root, exist_ok=True)

    torch.backends.cudnn.benchmark = True
    with ThreadPoolExecutor(8) as pool, torch.no_grad():
        if args.reference:
            write_reference(args, pool)
        if args.row:
            assert args.opt, '--row needs -opt'
            write_row(args, pool)


if __name__ == '__main__':
    main()
