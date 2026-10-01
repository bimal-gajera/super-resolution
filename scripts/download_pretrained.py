"""Download the official pretrained weights for the ``*_pretrained`` configs into experiments/pretrained_models/
(~360 MB). Checks the SHA-256 of every download, converts the keys to the BasicSR / srbench layout and saves
``{'params': state_dict}``. Skips existing files.

    python scripts/download_pretrained.py

| saved as                                      | model                        | source (authors' release)            |
|-----------------------------------------------|------------------------------|--------------------------------------|
| ESRGAN_PSNR_SRx4_DF2K_official.pth            | RRDBNet ×4, PSNR-oriented    | ESRGAN repo, RRDB_PSNR_x4.pth (Drive) |
| EDSR_Lx4_f256b32_DIV2K_official.pth           | EDSR ×4 (32 × 256)           | EDSR-PyTorch, edsr_x4                 |
| EDSR_Mx4_f64b16_DIV2K_official.pth            | EDSR baseline ×4 (16 × 64)   | EDSR-PyTorch, edsr_baseline_x4        |
| 001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth   | SwinIR-M classical SR ×4     | SwinIR release v0.0 (no conversion)   |
| 001_classicalSR_DIV2K_s48w8_SwinIR-M_x8.pth   | SwinIR-M classical SR ×8     | SwinIR release v0.0 (no conversion)   |

All are RGB models trained on bicubic-downsampled DIV2K / DF2K. How they are loaded into the 12-band networks:
docs/02_methods.md §2.2 ("Initialisation").
"""
import argparse
import os
import tempfile
from os import path as osp

import torch
from torch.hub import download_url_to_file

ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
GDRIVE = 'https://drive.usercontent.google.com/download?id={}&export=download&confirm=t'
SWINIR = 'https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/'
EDSR = 'https://cv.snu.ac.kr/research/EDSR/models/'


def convert_rrdb(sd):
    """ESRGAN repo (RRDBNet_arch.py) keys -> BasicSR RRDBNet keys."""
    rename = [('RRDB_trunk.', 'body.'), ('.RDB', '.rdb'), ('trunk_conv.', 'conv_body.'), ('upconv', 'conv_up'),
              ('HRconv.', 'conv_hr.')]
    out = {}
    for k, v in sd.items():
        for a, b in rename:
            k = k.replace(a, b)
        out[k] = v
    return out


def convert_edsr(sd):
    """EDSR-PyTorch keys -> BasicSR EDSR keys. Its sub_mean / add_mean layers hold the fixed DIV2K mean shift that
    BasicSR applies as ``rgb_mean`` (same values), so they are dropped."""
    last = max(int(k.split('.')[1]) for k in sd if k.startswith('body.'))  # body.<last> = conv after the blocks
    out = {}
    for k, v in sd.items():
        p = k.split('.')
        if p[0] in ('sub_mean', 'add_mean'):
            continue
        if p[0] == 'head':  # head.0.weight
            k = f'conv_first.{p[-1]}'
        elif p[0] == 'body' and int(p[1]) == last:  # body.32.weight
            k = f'conv_after_body.{p[-1]}'
        elif p[0] == 'body':  # body.<i>.body.{0,2}.weight
            k = f'body.{p[1]}.conv{1 if p[3] == "0" else 2}.{p[-1]}'
        elif p[:2] == ['tail', '0']:  # tail.0.<j>.weight: upsampler
            k = f'upsample.{p[2]}.{p[-1]}'
        elif p[:2] == ['tail', '1']:
            k = f'conv_last.{p[-1]}'
        out[k] = v
    return out


# saved name: (url, sha256 of the download, key conversion or None)
WEIGHTS = {
    'ESRGAN_PSNR_SRx4_DF2K_official.pth':
    (GDRIVE.format('1pJ_T-V1dpb1ewoEra1TGSWl5e6H7M4NN'),
     'f372b59f22929e1bc83fa58d78215c96f976de3b2eaeee736da1b348913da6cc', convert_rrdb),
    'EDSR_Lx4_f256b32_DIV2K_official.pth':
    (EDSR + 'edsr_x4-4f62e9ef.pt', '4f62e9ef1a4ec6a7d3da4ed837116cec641e4e1566f98187151d813e83a0c1c8', convert_edsr),
    'EDSR_Mx4_f64b16_DIV2K_official.pth':
    (EDSR + 'edsr_baseline_x4-6b446fab.pt', '6b446fab734f4de74448d2fd1f3f990f5bae726e49dc7eff2ae9cefe444a1723',
     convert_edsr),
    '001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth':
    (SWINIR + '001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth',
     '129dc773ba2d4c07f3eb0bb116fbe692011b7cc072d9ca12797cd3748198610a', None),
    '001_classicalSR_DIV2K_s48w8_SwinIR-M_x8.pth':
    (SWINIR + '001_classicalSR_DIV2K_s48w8_SwinIR-M_x8.pth',
     '3c3ae9238a7125e52aab457df052aa8981adb2fb8e42b6d4112a96b3c19432d7', None),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', default=osp.join(ROOT, 'experiments', 'pretrained_models'))
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    for name, (url, sha256, convert) in WEIGHTS.items():
        dst = osp.join(args.out, name)
        if osp.exists(dst):
            print(f'exists: {dst}')
            continue
        with tempfile.TemporaryDirectory(dir=args.out) as tmp:
            raw = osp.join(tmp, 'download')
            download_url_to_file(url, raw, hash_prefix=sha256, progress=False)
            if convert is None:
                os.replace(raw, dst)
            else:
                sd = torch.load(raw, map_location='cpu', weights_only=True)
                torch.save({'params': convert(sd)}, osp.join(tmp, name))
                os.replace(osp.join(tmp, name), dst)
        print(f'downloaded: {dst}')


if __name__ == '__main__':
    main()
