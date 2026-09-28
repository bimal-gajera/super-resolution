"""Prepare the Sentinel-2 -> Maxar super-resolution benchmark data.

Source layout (``--src``)::

    Maxar/<key>.tif        1584 x 1584 x 3 uint8 RGB, 0.303 m GSD
    Sentinel-2/<key'>.tif  48 x 48 x 12 uint16 L2A reflectance (x10000), 10 m GSD; key' = key with MAXAR -> S2

where ``key`` is the Maxar file stem, e.g. ``1030010076672300_120200222032_0011_MAXAR`` or
``103001011D651500_122002301303_MAXAR_0109`` (both naming schemes occur; the second token is a level-12 quadkey).

Stages (all written to ``--out``):

1. ``keys.txt``            all pair keys (sorted); row i of every cache array belongs to line i.
2. ``lq_s2.npy``           (N, 48, 48, 12) uint16 raw S2 stacks.
   ``gt_x{s}.npy``         (N, 48s, 48s, 3) uint8 Maxar tiles area-downsampled to 48*s px, for each ``--scales``.
   ``qa.csv``              per-pair QA (nodata, S2<->Maxar correlation, texture, brightness, residual shift).
   Caching is resumable: rerunning the script skips rows already listed in qa.csv.
3. ``meta_info/{train,val,test}.txt``  spatially disjoint split: whole quadkey groups (prefix of length
   ``--split-level``) are assigned to one split, so overlapping / repeated acquisitions never leak across splits.
   ``meta_info/*_clean.txt``  the same lists after the QA filter (see ``is_clean``).
4. ``stats.json``          per-band S2 mean/std on train_clean (used for input z-scoring) + summary.

Stages 3-4 only need qa.csv; rerun them with other thresholds via ``--lists-only``.

Example::

    python scripts/prepare_s2maxar.py --src /ocean/projects/cis250179p/purohit/Bimal/visual_small \\
        --out datasets/s2maxar --scales 4 8 --workers 5
"""
import argparse
import csv
import json
import os
import os.path as osp
import random
import re
import sys
import time
from collections import defaultdict
import multiprocessing as mp

import numpy as np

LR_SIZE, LR_BANDS, HR_NATIVE = 48, 12, 1584
KEY_RE = re.compile(r'^([0-9A-F]{16})_([0-3]{12})_(?:\d{4}_MAXAR|MAXAR_\d{4})$')
QA_FIELDS = [
    'key', 'row', 'ok', 'error', 'maxar_nodata', 's2_nodata', 'corr', 'hr_cv', 'lr_cv', 'b02_min', 'b02_p50',
    'b02_p99', 'shift_x', 'shift_y', 'shift_resp'
]


def s2_name(key):
    return key.replace('MAXAR', 'S2') + '.tif'


def quadkey(key):
    return KEY_RE.match(key).group(2)


# ----------------------------------------------------------------------------------------------------------------
# stage 2: caching + QA (worker side)
# ----------------------------------------------------------------------------------------------------------------
_W = {}


def _init_worker(src, out, scales, offsets):
    import cv2
    cv2.setNumThreads(1)
    _W.update(src=src, scales=scales, offsets=offsets)
    _W['fds'] = {name: os.open(osp.join(out, name), os.O_WRONLY) for name in offsets}


def _qa(hr, lr):
    """Cheap per-pair quality indicators computed at S2 resolution."""
    import cv2
    lr = lr.astype(np.float32)
    hr48 = cv2.resize(hr, (LR_SIZE, LR_SIZE), interpolation=cv2.INTER_AREA).astype(np.float32).mean(-1)
    lr_gray = lr[..., [3, 2, 1]].mean(-1)
    hr_cv = float(hr48.std() / (hr48.mean() + 1e-6))
    lr_cv = float(lr_gray.std() / (lr_gray.mean() + 1e-6))
    corr = float(np.corrcoef(hr48.ravel(), lr_gray.ravel())[0, 1]) if hr48.std() > 0 and lr_gray.std() > 0 else 0.
    # DEPRECATED shift estimate: both greys share the 48 px grid, so the phase-correlation peak locks to whole S2
    # pixels (a 0.5 px shift reads as 0). Kept for qa.csv compatibility; use scripts/registration.py instead.
    a = cv2.resize(hr48, (192, 192), interpolation=cv2.INTER_CUBIC)
    b = cv2.resize(lr_gray, (192, 192), interpolation=cv2.INTER_CUBIC)
    a = (a - a.mean()) / (a.std() + 1e-6)
    b = (b - b.mean()) / (b.std() + 1e-6)
    (dx, dy), resp = cv2.phaseCorrelate(a, b, cv2.createHanningWindow((192, 192), cv2.CV_32F))
    b02 = lr[..., 1]
    return dict(
        maxar_nodata=float((hr.max(-1) == 0).mean()),
        s2_nodata=float((lr.max(-1) == 0).mean()),
        corr=corr,
        hr_cv=hr_cv,
        lr_cv=lr_cv,
        b02_min=float(b02.min()),
        b02_p50=float(np.median(b02)),
        b02_p99=float(np.percentile(b02, 99)),
        shift_x=dx / 4,
        shift_y=dy / 4,
        shift_resp=float(resp))


def _process(job):
    import cv2
    import tifffile
    row, key = job
    rec = {'key': key, 'row': row, 'ok': 0, 'error': ''}
    try:
        hr = tifffile.imread(osp.join(_W['src'], 'Maxar', key + '.tif'))
        lr = tifffile.imread(osp.join(_W['src'], 'Sentinel-2', s2_name(key)))
        if hr.shape != (HR_NATIVE, HR_NATIVE, 3) or hr.dtype != np.uint8:
            raise ValueError(f'unexpected Maxar shape/dtype {hr.shape} {hr.dtype}')
        if lr.shape != (LR_SIZE, LR_SIZE, LR_BANDS) or lr.dtype != np.uint16:
            raise ValueError(f'unexpected S2 shape/dtype {lr.shape} {lr.dtype}')
        arrays = {'lq_s2.npy': lr}
        for s in _W['scales']:
            arrays[f'gt_x{s}.npy'] = cv2.resize(hr, (LR_SIZE * s, LR_SIZE * s), interpolation=cv2.INTER_AREA)
        for name, arr in arrays.items():  # pwrite: no memmap flushing issues when the job is killed
            buf = np.ascontiguousarray(arr).tobytes()
            os.pwrite(_W['fds'][name], buf, _W['offsets'][name] + row * len(buf))
        rec.update(_qa(hr, lr))
        rec['ok'] = 1
    except Exception as e:  # noqa: BLE001 - record and continue
        rec['error'] = f'{type(e).__name__}: {e}'.replace(',', ';').replace('\n', ' ')
    return rec


def build_cache(args, keys):
    out = args.out
    arrays = {'lq_s2.npy': ((LR_SIZE, LR_SIZE, LR_BANDS), np.uint16)}
    for s in args.scales:
        arrays[f'gt_x{s}.npy'] = ((LR_SIZE * s, LR_SIZE * s, 3), np.uint8)
    offsets = {}
    for name, (shape, dtype) in arrays.items():
        path, full = osp.join(out, name), (len(keys), ) + shape
        if osp.exists(path):
            mm = np.load(path, mmap_mode='r')
            if mm.shape != full or mm.dtype != dtype:
                sys.exit(f'{path} exists with shape {mm.shape}, expected {full}; remove it (or use another --out)')
        else:
            mm = np.lib.format.open_memmap(path, mode='w+', dtype=dtype, shape=full)
        offsets[name] = mm.offset
        del mm

    qa_path, scales_path = osp.join(out, 'qa.csv'), osp.join(out, 'cache_scales.json')
    prev_scales = None
    if osp.exists(scales_path):
        with open(scales_path) as f:
            prev_scales = json.load(f)
    done = set()
    if osp.exists(qa_path):
        if prev_scales == sorted(args.scales):  # resume an interrupted run
            # keep complete, successful rows only (a killed run can leave a truncated last line; errors are retried)
            keyset, rows = set(keys), {}
            with open(qa_path) as f:
                for r in csv.DictReader(f):
                    if None not in r.values() and r['key'] in keyset and r['ok'] == '1':
                        rows[r['key']] = r
            with open(qa_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=QA_FIELDS)
                writer.writeheader()
                writer.writerows(rows.values())
            done = set(rows)
        else:  # different scale set -> every row has to be (re)written
            print(f'scales changed ({prev_scales} -> {sorted(args.scales)}): recomputing all rows', flush=True)
            os.replace(qa_path, qa_path + '.old')
    with open(scales_path, 'w') as f:
        json.dump(sorted(args.scales), f)
    todo = [(i, k) for i, k in enumerate(keys) if k not in done]
    print(f'cache: {len(keys)} pairs, {len(done)} already done, {len(todo)} to process with {args.workers} workers',
          flush=True)
    if not todo:
        return
    write_header = not osp.exists(qa_path)
    t0, n_err = time.time(), 0
    # 'spawn': forked workers can deadlock if the parent already used OpenCV/OpenMP thread pools
    ctx = mp.get_context('spawn')
    with open(qa_path, 'a', newline='') as f, ctx.Pool(args.workers, _init_worker,
                                                      (args.src, out, args.scales, offsets)) as pool:
        writer = csv.DictWriter(f, fieldnames=QA_FIELDS)
        if write_header:
            writer.writeheader()
        for i, rec in enumerate(pool.imap_unordered(_process, todo, chunksize=8), 1):
            writer.writerow({k: (f'{v:.5g}' if isinstance(v, float) else v) for k, v in rec.items()})
            n_err += not rec['ok']
            if i % 500 == 0 or i == len(todo):
                f.flush()
                rate = i / (time.time() - t0)
                print(f'  {i}/{len(todo)}  {rate:.1f} pairs/s  eta {(len(todo) - i) / rate / 60:.1f} min  '
                      f'errors {n_err}', flush=True)


# ----------------------------------------------------------------------------------------------------------------
# stage 3-4: split, QA filter, stats
# ----------------------------------------------------------------------------------------------------------------
def spatial_split(keys, level, ratios, seed):
    groups = defaultdict(list)
    for k in keys:
        groups[quadkey(k)[:level]].append(k)
    names = sorted(groups)
    random.Random(seed).shuffle(names)
    n = len(keys)
    target = {'val': ratios[1] * n, 'test': ratios[2] * n}
    split = {'train': [], 'val': [], 'test': []}
    for g in names:
        size = len(groups[g])
        dest = 'train'
        for s in ('val', 'test'):
            if len(split[s]) + size <= target[s]:
                dest = s
                break
        split[dest].extend(groups[g])
    return {s: sorted(v) for s, v in split.items()}, {g: s for s in split for g in {quadkey(k)[:level] for k in split[s]}}


def is_clean(r, args):
    """QA filter: drop unreadable pairs, nodata, and pairs whose S2 content does not match Maxar (mostly clouds /
    haze / snow in S2 or strong temporal change). Flat tiles (water, uniform fields) have meaningless correlation
    and are kept when both images are flat."""
    if not int(r['ok']):
        return False
    if float(r['maxar_nodata']) > args.max_nodata or float(r['s2_nodata']) > args.max_nodata:
        return False
    if float(r['b02_p50']) > args.max_b02:
        return False
    flat = float(r['hr_cv']) < args.flat_cv and float(r['lr_cv']) < args.flat_cv
    return float(r['corr']) >= args.min_corr or flat


def write_lists(args, keys):
    out = args.out
    with open(osp.join(out, 'qa.csv')) as f:
        qa = {r['key']: r for r in csv.DictReader(f)}
    missing = [k for k in keys if k not in qa]
    if missing:
        sys.exit(f'qa.csv misses {len(missing)} keys -> caching did not finish; rerun without --lists-only')
    ok_keys = [k for k in keys if int(qa[k]['ok'])]
    split, group2split = spatial_split(ok_keys, args.split_level, args.split_ratios, args.seed)
    os.makedirs(osp.join(out, 'meta_info'), exist_ok=True)
    summary = {'n_pairs': len(keys), 'n_unreadable': len(keys) - len(ok_keys), 'split_level': args.split_level,
               'qa_filter': {k: getattr(args, k) for k in ('min_corr', 'flat_cv', 'max_nodata', 'max_b02')},
               'splits': {}}
    for s, lst in split.items():
        clean = [k for k in lst if is_clean(qa[k], args)]
        for name, l in ((s, lst), (f'{s}_clean', clean)):
            with open(osp.join(out, 'meta_info', f'{name}.txt'), 'w') as f:
                f.write('\n'.join(l) + '\n')
        summary['splits'][s] = {'all': len(lst), 'clean': len(clean),
                                'groups': sum(1 for v in group2split.values() if v == s)}
    with open(osp.join(out, 'meta_info', 'split_groups.json'), 'w') as f:
        json.dump(group2split, f, indent=0, sort_keys=True)

    # input normalisation statistics on (a sample of) train_clean
    lq = np.load(osp.join(out, 'lq_s2.npy'), mmap_mode='r')
    row = {k: i for i, k in enumerate(keys)}
    train_clean = [k for k in split['train'] if is_clean(qa[k], args)]
    sample = sorted(random.Random(args.seed).sample(train_clean, min(len(train_clean), 5000)))
    x = np.stack([lq[row[k]] for k in sample]).reshape(-1, LR_BANDS).astype(np.float64)
    summary['s2_bands'] = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']
    summary['s2_mean'] = x.mean(0).round(3).tolist()
    summary['s2_std'] = x.std(0).round(3).tolist()
    summary['stats_from'] = f'{len(sample)} train_clean tiles'
    with open(osp.join(out, 'stats.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k not in ('s2_mean', 's2_std', 's2_bands')}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--src', default='/ocean/projects/cis250179p/purohit/Bimal/visual_small')
    parser.add_argument('--out', default='datasets/s2maxar')
    parser.add_argument('--scales', type=int, nargs='+', default=[4, 8], help='HR targets to cache (48*s px)')
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument('--limit', type=int, default=None, help='random subset of N pairs (for debugging)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--split-level', type=int, default=10, help='quadkey prefix length that defines a group')
    parser.add_argument('--split-ratios', type=float, nargs=3, default=[0.9, 0.05, 0.05])
    parser.add_argument('--min-corr', type=float, default=0.2, help='min S2<->Maxar gray correlation')
    parser.add_argument('--flat-cv', type=float, default=0.1, help='both tiles flatter than this: keep anyway')
    parser.add_argument('--max-nodata', type=float, default=0.05)
    parser.add_argument('--max-b02', type=float, default=4000, help='median S2 B02 above this: cloud/snow')
    parser.add_argument('--lists-only', action='store_true', help='skip caching, only redo split/QA lists/stats')
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    keys_path = osp.join(args.out, 'keys.txt')
    if osp.exists(keys_path):
        with open(keys_path) as f:
            keys = [line.strip() for line in f if line.strip()]
    else:
        t0 = time.time()
        maxar = {f[:-4] for f in os.listdir(osp.join(args.src, 'Maxar')) if f.endswith('.tif')}
        s2 = set(os.listdir(osp.join(args.src, 'Sentinel-2')))
        keys = sorted(k for k in maxar if KEY_RE.match(k) and s2_name(k) in s2)
        print(f'listed {len(maxar)} Maxar files, {len(keys)} valid pairs ({time.time() - t0:.0f}s)')
        if args.limit:
            keys = sorted(random.Random(args.seed).sample(keys, args.limit))
        with open(keys_path, 'w') as f:
            f.write('\n'.join(keys) + '\n')

    if not args.lists_only:
        build_cache(args, keys)
    write_lists(args, keys)


if __name__ == '__main__':
    main()
