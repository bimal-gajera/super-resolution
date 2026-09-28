"""Sub-pixel S2 <-> Maxar registration check (forward model, no interpolation artefacts).

For every candidate offset on a 1/8 S2-pixel grid (1.25 m), the Maxar tile at 8 px per S2 pixel (``gt_x8``) is
box-averaged into 8 x 8 blocks (an S2 pixel footprint) and correlated with the S2 grey image (B04+B03+B02).
The offset with the highest correlation is the registration estimate.

Sign convention: (dx, dy) = position in the Maxar grid (in S2 pixels) of the content seen at S2 pixel (i, j),
minus (i, j); x = east, y = south. With perfect georeferencing the tiles' origins alone predict
(dx, dy) = (-0.364, -0.364) (the Maxar tile origin lies ~3.6 m east and south of the S2 tile origin).

    python scripts/registration.py --root datasets/s2maxar --workers 5          # all pairs -> registration.csv
"""
import argparse
import csv
import os.path as osp
from multiprocessing import get_context

import numpy as np

R = 16  # search radius in x8 pixels (= 2 S2 px)
B = 2   # S2 border pixels ignored (room for the search)


def estimate_offset(hr8_gray, lr_gray):
    """hr8_gray: (384, 384) float, lr_gray: (48, 48) float -> (dx, dy, best_corr, corr_at_zero)."""
    n = lr_gray.shape[0] - 2 * B
    ii = np.zeros((hr8_gray.shape[0] + 1, hr8_gray.shape[1] + 1))
    ii[1:, 1:] = hr8_gray.cumsum(0).cumsum(1)
    offs = np.arange(-R, R + 1)
    base = 8 * np.arange(B, B + n)
    rows = (base[None, :] + offs[:, None])  # (O, n)
    cols = rows
    r0, r1 = rows[:, None, :, None], rows[:, None, :, None] + 8
    c0, c1 = cols[None, :, None, :], cols[None, :, None, :] + 8
    blocks = ii[r1, c1] - ii[r0, c1] - ii[r1, c0] + ii[r0, c0]  # (Oy, Ox, n, n)
    t = lr_gray[B:B + n, B:B + n]
    t = (t - t.mean()) / (t.std() + 1e-9)
    b = blocks.reshape(len(offs), len(offs), -1)
    b = (b - b.mean(-1, keepdims=True)) / (b.std(-1, keepdims=True) + 1e-9)
    corr = (b * t.reshape(1, 1, -1)).mean(-1)
    iy, ix = np.unravel_index(np.argmax(corr), corr.shape)
    return offs[ix] / 8, offs[iy] / 8, float(corr[iy, ix]), float(corr[R, R])


def _work(args):
    root, rows = args
    lq = np.load(osp.join(root, 'lq_s2.npy'), mmap_mode='r')
    g8 = np.load(osp.join(root, 'gt_x8.npy'), mmap_mode='r')
    out = []
    for r in rows:
        hr = np.asarray(g8[r], np.float64).mean(-1)
        lr = np.asarray(lq[r][..., [3, 2, 1]], np.float64).mean(-1)
        out.append((r, ) + estimate_offset(hr, lr))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='datasets/s2maxar')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--limit', type=int, default=None)
    a = p.parse_args()
    keys = [line.strip() for line in open(osp.join(a.root, 'keys.txt'))]
    rows = list(range(len(keys)))
    if a.limit:
        rows = sorted(np.random.default_rng(0).choice(rows, a.limit, replace=False).tolist())
    chunks = [(a.root, rows[i:i + 200]) for i in range(0, len(rows), 200)]
    with get_context('spawn').Pool(a.workers) as pool, open(osp.join(a.root, 'registration.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['key', 'reg_dx', 'reg_dy', 'reg_corr', 'corr_at_zero'])
        for i, res in enumerate(pool.imap(_work, chunks), 1):
            for r, dx, dy, c, c0 in res:
                w.writerow([keys[r], dx, dy, f'{c:.4f}', f'{c0:.4f}'])
            if i % 25 == 0:
                print(f'{i * 200}/{len(rows)}', flush=True)


if __name__ == '__main__':
    main()
