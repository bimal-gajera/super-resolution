"""Download the weights AdcSR needs into experiments/pretrained_models/ (~17 GB). Resumable; skips existing files.

    envs/adcsr/bin/python scripts/download_adcsr_weights.py

* experiments/pretrained_models/AdcSR/: from Hugging Face ``Guaishou74851/AdcSR``
    halfDecoder.ckpt        stage-1 half-width VAE decoder (student tail, frozen)
    osediff.pkl             OSEDiff teacher (SD 2.1 VAE + U-Net with merged LoRA)
    ram_swin_large_14m.pth  RAM tagger  } DAPE prompt extractor (text conditioning of teacher and discriminator)
    DAPE.pth                DAPE LoRA   }
    net_params_200.pkl      released AdcSR student (zero-shot reference: options/test/AdcSR/test_AdcSR_x4_official.yml)
* experiments/pretrained_models/stable-diffusion-2-1-base/: diffusers-format SD 2.1-base (U-Net, VAE, text encoder,
  tokenizer, scheduler) from the ``Manojb/stable-diffusion-2-1-base`` mirror (stabilityai/stable-diffusion-2-1-base
  is no longer public);
* the ``bert-base-uncased`` tokenizer used by RAM (Hugging Face cache).
"""
import argparse
import os
import shutil
from os import path as osp

from huggingface_hub import hf_hub_download, snapshot_download

ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
ADCSR_REPO = 'Guaishou74851/AdcSR'
ADCSR_FILES = [
    'weight/pretrained/halfDecoder.ckpt',
    'weight/pretrained/osediff.pkl',
    'weight/pretrained/ram_swin_large_14m.pth',
    'weight/pretrained/DAPE.pth',
    'weight/net_params_200.pkl',
]
SD_REPO = 'Manojb/stable-diffusion-2-1-base'
SD_FILES = [
    'model_index.json', 'scheduler/*', 'tokenizer/*', 'feature_extractor/*', 'text_encoder/config.json',
    'text_encoder/model.safetensors', 'unet/config.json', 'unet/diffusion_pytorch_model.safetensors', 'vae/config.json',
    'vae/diffusion_pytorch_model.safetensors'
]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', default=osp.join(ROOT, 'experiments', 'pretrained_models'))
    parser.add_argument('--skip-official', action='store_true', help='do not download net_params_200.pkl (1.8 GB)')
    args = parser.parse_args()

    adcsr_dir = osp.join(args.out, 'AdcSR')
    os.makedirs(adcsr_dir, exist_ok=True)
    for name in ADCSR_FILES:
        if args.skip_official and name.endswith('net_params_200.pkl'):
            continue
        dst = osp.join(adcsr_dir, osp.basename(name))
        if osp.exists(dst):
            print(f'exists: {dst}')
            continue
        src = hf_hub_download(ADCSR_REPO, name, local_dir=osp.join(adcsr_dir, '.hf'))
        os.replace(src, dst)
        print(f'downloaded: {dst}')
    shutil.rmtree(osp.join(adcsr_dir, '.hf'), ignore_errors=True)  # download metadata

    sd_dir = osp.join(args.out, 'stable-diffusion-2-1-base')
    snapshot_download(SD_REPO, local_dir=sd_dir, allow_patterns=SD_FILES)
    print(f'downloaded: {sd_dir}')

    from transformers import BertTokenizer
    BertTokenizer.from_pretrained('bert-base-uncased')
    print('cached: bert-base-uncased tokenizer')


if __name__ == '__main__':
    main()
