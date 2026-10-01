import json
import os.path as osp
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils import data as data

from srbench.data.transforms import paired_random_crop
from srbench.utils.registry import DATASET_REGISTRY

# Band order of the 12-band Sentinel-2 L2A stacks (B10 is not part of L2A)
S2_BANDS = ('B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12')
S2_RGB = (3, 2, 1)  # B04, B03, B02
S2_TILE = 48  # LR tile size (48 px @ 10 m = 480 m)
S2_REFLECTANCE_SCALE = 10000.


def read_keys(path):
    with open(path) as f:
        return [line.strip().split()[0] for line in f if line.strip() and not line.startswith('#')]


def s2_to_rgb_vis(lq_raw):
    """Percentile-stretched B04/B03/B02 composite in [0, 1] (for visualisation only)."""
    rgb = lq_raw[..., list(S2_RGB)].astype(np.float32)
    lo, hi = np.percentile(rgb, 1), np.percentile(rgb, 99)
    return np.clip((rgb - lo) / max(hi - lo, 1e-6), 0, 1)


def _augment(imgs, hflip, vflip, rot90):
    out = []
    for img in imgs:
        if hflip:
            img = img[:, ::-1]
        if vflip:
            img = img[::-1]
        if rot90:
            img = img.transpose(1, 0, 2)
        out.append(np.ascontiguousarray(img))
    return out


def _to_tensor(img):
    return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).float()


@DATASET_REGISTRY.register()
class S2MaxarDataset(data.Dataset):
    """Paired Sentinel-2 (LR, 10 m, 12 bands) -> Maxar (HR, RGB) dataset.

    The HR target is the Maxar tile (0.303 m) area-downsampled to ``48 * scale`` pixels, i.e. 2.5 m for x4, 1.25 m
    for x8, 0.625 m for x16 and 0.3125 m for x32. Both backends produce identical samples.

    Args:
        opt (dict): Config for the dataset. It contains the following keys:
            dataroot (str): Output folder of ``scripts/prepare_s2maxar.py`` (``keys.txt``, ``lq_s2.npy``,
                ``gt_x{scale}.npy``, ``stats.json``).
            meta_info_file (str): Text file with one pair key per line (e.g. ``meta_info/train_clean.txt``).
            io_backend (dict): ``{type: npy}`` (default, memory-mapped cache) or
                ``{type: disk, maxar_dir: ..., s2_dir: ...}`` to read and resize the raw GeoTIFFs on the fly.
            scale (int): Upsampling factor (injected from the top-level ``scale``).
            lq_source (str): ``s2`` (default) or ``bicubic``: synthetic LR = bicubic-downsampled GT (3 bands),
                useful as a pipeline sanity check.
            lq_bands (list[int] | None): Indices into the 12-band stack (``S2_BANDS``). Default: all 12.
            lq_norm (str): ``meanstd`` (per-band z-score with ``stats.json``, default) or ``reflectance`` (/10000).
            gt_size (int | None): HR crop size for training. Default: None (full tile).
            use_hflip (bool), use_rot (bool): Flip / rot90 augmentation (train phase only).
            max_samples (int | None): Evenly subsample the list to at most this many pairs (e.g. fast validation).
            return_vis (bool): Also return ``lq_rgb`` (stretched S2 RGB) for visualisation. Default: True for
                val/test.
            phase (str): 'train' | 'val' | 'test' (injected by the option parser).
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.scale = opt['scale']
        self.phase = opt.get('phase', 'train')
        self.io_opt = dict(opt.get('io_backend', {'type': 'npy'}))
        self.backend = self.io_opt.get('type', 'npy')
        self.dataroot = opt.get('dataroot')
        self.lq_source = opt.get('lq_source', 's2')
        self.lq_norm = opt.get('lq_norm', 'meanstd')
        self.gt_size = opt.get('gt_size')
        self.return_vis = opt.get('return_vis', self.phase != 'train')
        assert self.backend in ('npy', 'disk'), f'unknown io_backend {self.backend}'
        assert self.lq_source in ('s2', 'bicubic'), f'unknown lq_source {self.lq_source}'
        assert self.lq_norm in ('meanstd', 'reflectance'), f'unknown lq_norm {self.lq_norm}'

        self.keys = read_keys(opt['meta_info_file'])
        max_samples = opt.get('max_samples')
        if max_samples and len(self.keys) > max_samples:
            idx = np.linspace(0, len(self.keys) - 1, max_samples).round().astype(int)
            self.keys = [self.keys[i] for i in idx]

        bands = opt.get('lq_bands')
        self.lq_bands = list(range(len(S2_BANDS))) if bands is None else list(bands)

        stats_file = opt.get('stats_file') or (osp.join(self.dataroot, 'stats.json') if self.dataroot else None)
        if self.lq_source == 's2' and self.lq_norm == 'meanstd':
            with open(stats_file) as f:
                stats = json.load(f)
            self.lq_mean = np.asarray(stats['s2_mean'], np.float32)[self.lq_bands]
            self.lq_std = np.asarray(stats['s2_std'], np.float32)[self.lq_bands]

        if self.backend == 'npy':
            all_keys = read_keys(osp.join(self.dataroot, 'keys.txt'))
            key2row = {k: i for i, k in enumerate(all_keys)}
            missing = [k for k in self.keys if k not in key2row]
            if missing:
                raise KeyError(f'{len(missing)} keys of {opt["meta_info_file"]} are not in the cache, e.g. {missing[0]}')
            self.rows = np.asarray([key2row[k] for k in self.keys], dtype=np.int64)
            self.lq_file = osp.join(self.dataroot, 'lq_s2.npy')
            self.gt_file = osp.join(self.dataroot, f'gt_x{self.scale}.npy')
            if not osp.exists(self.gt_file):
                raise FileNotFoundError(f'{self.gt_file} not found: run scripts/prepare_s2maxar.py --add-scales {self.scale}')
        self._lq = self._gt = None  # memmaps are opened lazily in each dataloader worker

    @property
    def num_in_ch(self):
        return 3 if self.lq_source == 'bicubic' else len(self.lq_bands)

    def __len__(self):
        return len(self.keys)

    def _load_pair(self, index):
        """Returns raw LR (48, 48, 12) uint16 and HR (48*scale, 48*scale, 3) uint8 arrays."""
        if self.backend == 'npy':
            if self._lq is None:
                self._lq = np.load(self.lq_file, mmap_mode='r')
                self._gt = np.load(self.gt_file, mmap_mode='r')
            row = self.rows[index]
            return np.array(self._lq[row]), np.array(self._gt[row])
        # raw GeoTIFFs
        import cv2
        import tifffile
        key = self.keys[index]
        gt = tifffile.imread(osp.join(self.io_opt['maxar_dir'], key + '.tif'))
        lq = tifffile.imread(osp.join(self.io_opt['s2_dir'], key.replace('MAXAR', 'S2') + '.tif'))
        size = lq.shape[0] * self.scale
        gt = cv2.resize(gt, (size, size), interpolation=cv2.INTER_AREA)
        return lq, gt

    def __getitem__(self, index):
        key = self.keys[index]
        lq_raw, gt = self._load_pair(index)
        img_gt = gt.astype(np.float32) / 255.

        if self.lq_source == 'bicubic':
            t = torch.from_numpy(img_gt.transpose(2, 0, 1)).unsqueeze(0)
            h, w = img_gt.shape[0] // self.scale, img_gt.shape[1] // self.scale
            t = F.interpolate(t, size=(h, w), mode='bicubic', align_corners=False, antialias=True)
            img_lq = t.clamp(0, 1)[0].numpy().transpose(1, 2, 0)
        else:
            img_lq = lq_raw[..., self.lq_bands].astype(np.float32)
            if self.lq_norm == 'meanstd':
                img_lq = (img_lq - self.lq_mean) / self.lq_std
            else:
                img_lq = img_lq / S2_REFLECTANCE_SCALE

        vis = [s2_to_rgb_vis(lq_raw)] if self.return_vis else []
        if self.phase == 'train':
            if self.gt_size is not None and self.gt_size < img_gt.shape[0]:
                if vis:  # crop the visualisation together with lq
                    img_gt, lqs = paired_random_crop(img_gt, [img_lq] + vis, self.gt_size, self.scale, key)
                    img_lq, vis = lqs[0], lqs[1:]
                else:
                    img_gt, img_lq = paired_random_crop(img_gt, img_lq, self.gt_size, self.scale, key)
            hflip = self.opt.get('use_hflip', False) and random.random() < 0.5
            vflip = self.opt.get('use_rot', False) and random.random() < 0.5
            rot90 = self.opt.get('use_rot', False) and random.random() < 0.5
            img_gt, img_lq, *vis = _augment([img_gt, img_lq] + vis, hflip, vflip, rot90)

        out = {'lq': _to_tensor(img_lq), 'gt': _to_tensor(img_gt), 'lq_path': key, 'gt_path': key, 'key': key}
        if vis:
            out['lq_rgb'] = _to_tensor(vis[0])
        return out
