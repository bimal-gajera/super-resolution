"""Fast checks on synthetic data (no GPU, no real dataset needed): python -m pytest tests"""
import json
import os
import sys

import cv2
import numpy as np
import pytest
import tifffile
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'scripts'))

import prepare_s2maxar as prep  # noqa: E402
from srbench.archs import build_network  # noqa: E402
from srbench.data.s2maxar_dataset import S2MaxarDataset  # noqa: E402
from srbench.metrics import calculate_metric  # noqa: E402

KEYS = ['1030010076672300_120200222032_0011_MAXAR', '1030010076672300_120200222032_MAXAR_0012',
        '10400100889ABF00_302000123001_0005_MAXAR']


@pytest.fixture(scope='module')
def data_root(tmp_path_factory):
    """Tiny raw dataset (GeoTIFF-like) + the cache built by the real prepare script."""
    root = tmp_path_factory.mktemp('s2maxar')
    src, out = root / 'src', root / 'cache'
    (src / 'Maxar').mkdir(parents=True)
    (src / 'Sentinel-2').mkdir()
    rng = np.random.default_rng(0)
    for k in KEYS:
        lr = rng.integers(200, 4000, (48, 48, 12), dtype=np.uint16)
        hr = cv2.resize(lr[..., [3, 2, 1]].astype(np.float32) / 16, (1584, 1584), interpolation=cv2.INTER_CUBIC)
        tifffile.imwrite(src / 'Maxar' / f'{k}.tif', np.clip(hr, 0, 255).astype(np.uint8), compression='lzw')
        tifffile.imwrite(src / 'Sentinel-2' / prep.s2_name(k), lr, compression='lzw')
    sys_argv = sys.argv
    sys.argv = ['prep', '--src', str(src), '--out', str(out), '--scales', '4', '--workers', '1',
                '--split-level', '1', '--split-ratios', '0.34', '0.33', '0.33']
    try:
        prep.main()
    finally:
        sys.argv = sys_argv
    return src, out


def _opt(out, **kw):
    opt = dict(name='t', type='S2MaxarDataset', dataroot=str(out), meta_info_file=str(out / 'keys.txt'),
               io_backend={'type': 'npy'}, scale=4, phase='val')
    opt.update(kw)
    return opt


def test_prepare_outputs(data_root):
    _, out = data_root
    assert (out / 'keys.txt').read_text().split() == sorted(KEYS)
    assert np.load(out / 'lq_s2.npy', mmap_mode='r').shape == (3, 48, 48, 12)
    assert np.load(out / 'gt_x4.npy', mmap_mode='r').shape == (3, 192, 192, 3)
    stats = json.loads((out / 'stats.json').read_text())
    assert len(stats['s2_mean']) == 12 and stats['n_unreadable'] == 0


def test_split_is_group_disjoint():
    keys = [f'1030010076672300_{q}_{i:04d}_MAXAR' for q in ('120200222032', '120200222033', '122000311200',
                                                            '302000123001', '213131010201') for i in range(4)]
    split, _ = prep.spatial_split(keys, level=10, ratios=[0.6, 0.2, 0.2], seed=0)
    groups = {s: {prep.quadkey(k)[:10] for k in v} for s, v in split.items()}
    assert not (groups['train'] & groups['val']) and not (groups['train'] & groups['test'])
    assert not (groups['val'] & groups['test'])
    assert sum(len(v) for v in split.values()) == len(keys)


def test_npy_and_disk_backends_match(data_root):
    src, out = data_root
    a = S2MaxarDataset(_opt(out))
    b = S2MaxarDataset(_opt(out, io_backend={'type': 'disk', 'maxar_dir': str(src / 'Maxar'),
                                             's2_dir': str(src / 'Sentinel-2')}))
    for i in range(len(a)):
        sa, sb = a[i], b[i]
        assert sa['key'] == sb['key']
        assert torch.equal(sa['lq'], sb['lq']) and torch.equal(sa['gt'], sb['gt'])
        assert sa['lq'].shape == (12, 48, 48) and sa['gt'].shape == (3, 192, 192) and 'lq_rgb' in sa


def test_train_augmentation_keeps_pairs_aligned(data_root):
    """Crop/flip/rot must be applied identically: GT area-downsampled back to LR size must still correlate
    with the (synthetic, GT-generating) LR bands."""
    _, out = data_root
    ds = S2MaxarDataset(_opt(out, phase='train', gt_size=96, use_hflip=True, use_rot=True, lq_norm='reflectance'))
    for i in range(12):
        s = ds[i % len(ds)]
        assert s['lq'].shape == (12, 24, 24) and s['gt'].shape == (3, 96, 96)
        gt_small = torch.nn.functional.avg_pool2d(s['gt'][None], 4)[0].mean(0).flatten()
        lq_rgb = s['lq'][[3, 2, 1]].mean(0).flatten()
        assert np.corrcoef(gt_small.numpy(), lq_rgb.numpy())[0, 1] > 0.9


def test_band_subset_and_bicubic_source(data_root):
    _, out = data_root
    s = S2MaxarDataset(_opt(out, lq_bands=[3, 2, 1]))[0]
    assert s['lq'].shape == (3, 48, 48)
    s = S2MaxarDataset(_opt(out, lq_source='bicubic'))[0]
    assert s['lq'].shape == (3, 48, 48) and 0 <= s['lq'].min() and s['lq'].max() <= 1


@pytest.mark.parametrize('scale', [4, 8])
def test_arch_shapes(scale):
    x = torch.randn(1, 12, 16, 16)
    nets = [
        dict(type='RRDBNet', num_in_ch=12, num_out_ch=3, scale=scale, num_feat=16, num_block=1, num_grow_ch=8),
        dict(type='SwinIR', upscale=scale, in_chans=12, out_chans=3, img_size=16, window_size=8, depths=[2],
             embed_dim=24, num_heads=[2], mlp_ratio=2, upsampler='pixelshuffle'),
        dict(type='EDSR', num_in_ch=12, num_out_ch=3, num_feat=16, num_block=2, upscale=scale, res_scale=0.1),
        dict(type='BicubicBaseline', num_in_ch=12, num_out_ch=3, scale=scale),
    ]
    for opt in nets:
        assert build_network(opt)(x).shape == (1, 3, 16 * scale, 16 * scale), opt['type']


def test_discriminator_sizes():
    for size in (128, 192, 256):
        d = build_network(dict(type='VGGStyleDiscriminator', num_in_ch=3, num_feat=8, input_size=size))
        assert d(torch.randn(2, 3, size, size)).shape == (2, 1)


def test_metrics_sanity():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    noisy = np.clip(img.astype(int) + rng.integers(-8, 9, img.shape), 0, 255).astype(np.uint8)
    data = {'img': noisy, 'img2': img}
    psnr = calculate_metric(data, {'type': 'calculate_psnr', 'crop_border': 4})
    assert 30 < psnr < 40
    assert calculate_metric({'img': img, 'img2': img}, {'type': 'calculate_ssim', 'crop_border': 4}) == pytest.approx(1)


ADCSR_WEIGHTS = os.path.join(REPO, 'experiments', 'pretrained_models')


@pytest.mark.skipif(not os.path.exists(os.path.join(ADCSR_WEIGHTS, 'AdcSR', 'net_params_200.pkl')),
                    reason='AdcSR weights not downloaded (scripts/download_adcsr_weights.py)')
def test_adcsr_matches_released_weights():
    """The ported student has exactly the parameter names/shapes of the released AdcSR model (envs/adcsr only)."""
    pytest.importorskip('diffusers')
    net = build_network(dict(type='AdcSR', sd_model=os.path.join(ADCSR_WEIGHTS, 'stable-diffusion-2-1-base'),
                             half_decoder=os.path.join(ADCSR_WEIGHTS, 'AdcSR', 'halfDecoder.ckpt'), sd_init=False))
    released = torch.load(os.path.join(ADCSR_WEIGHTS, 'AdcSR', 'net_params_200.pkl'), map_location='cpu',
                          weights_only=True)
    released = {k.removeprefix('module.'): v for k, v in released.items()}
    ours = net.state_dict()
    assert set(released) <= set(ours) and all(k.startswith('tail.') for k in set(ours) - set(released))
    assert all(ours[k].shape == v.shape for k, v in released.items())
    net.load_state_dict(released, strict=False)
    with torch.no_grad():
        out = net(torch.rand(1, 3, 16, 16))
    assert out.shape == (1, 3, 64, 64) and torch.isfinite(out).all()
