"""Plot training losses and validation metrics from tensorboard logs (tb_logger/<run>/).

Runs resumed across Slurm jobs have one event file per job; they are merged, and for iterations logged twice
(job A ran past its last checkpoint, job B redid them) the later job's value is kept.

    python scripts/plot_training.py smoke_RRDBNet_PSNR_x4 smoke_SwinIR_SRx4 smoke_ESRGAN_x4 \
        --baseline Bicubic_x4_S2Maxar_linearcolor_5k_B16G1 --out experiments/smoke_curves.png
"""
import argparse
import glob
import os.path as osp

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: E402

REPO = osp.dirname(osp.dirname(osp.abspath(__file__)))


def load_scalars(run):
    """{tag: {step: value}} merged over all event files of a run (later files win)."""
    out = {}
    for f in sorted(glob.glob(osp.join(REPO, 'tb_logger', run, 'events.out.tfevents.*')), key=osp.getmtime):
        acc = EventAccumulator(f, size_guidance={'scalars': 0})
        acc.Reload()
        for tag in acc.Tags()['scalars']:
            out.setdefault(tag, {}).update({e.step: e.value for e in acc.Scalars(tag)})
    return out


def series(scalars, tag):
    d = scalars.get(tag, {})
    steps = sorted(d)
    return steps, [d[s] for s in steps]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('runs', nargs='+')
    p.add_argument('--baseline', default=None, help='run drawn as a horizontal line (final validation value)')
    p.add_argument('--val', default='S2Maxar_val')
    p.add_argument('--out', default=osp.join(REPO, 'experiments', 'training_curves.png'))
    a = p.parse_args()

    data = {r: load_scalars(r) for r in a.runs}
    base = load_scalars(a.baseline) if a.baseline else None
    colors = dict(zip(a.runs, ['#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#ff7f0e']))

    panels = [('losses/l_pix', 'train L1 (mean / 100 it)'), ('losses/l_g_percep', 'train perceptual (GAN runs)'),
              (f'metrics/{a.val}/psnr', 'val PSNR (dB) ↑'), (f'metrics/{a.val}/ssim', 'val SSIM ↑'),
              (f'metrics/{a.val}/lpips', 'val LPIPS ↓'), ('losses/l_d_real', 'discriminator loss (real / fake)')]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    for ax, (tag, title) in zip(axes.flat, panels):
        for r in a.runs:
            s, v = series(data[r], tag)
            if s:
                ax.plot(s, v, marker='o' if 'metrics' in tag else None, ms=4, lw=1.6, color=colors[r], label=r)
            if tag == 'losses/l_d_real':
                s2, v2 = series(data[r], 'losses/l_d_fake')
                if s2:
                    ax.plot(s2, v2, ls='--', lw=1.2, color=colors[r], label=f'{r} (fake)')
        if base is not None and 'metrics' in tag:
            s, v = series(base, tag)
            if v:
                ax.axhline(v[-1], color='gray', ls=':', lw=1.5, label=f'{a.baseline.split("_")[0]} baseline ({v[-1]:.3f})')
        if tag == 'losses/l_pix' and base is not None:
            s, v = series(base, tag)
            if s:
                ax.plot(s, v, color='gray', lw=1, alpha=0.7, label=a.baseline.split('_')[0])
        ax.set_title(title)
        ax.set_xlabel('iteration')
        ax.grid(alpha=0.3)
        if ax.lines:
            ax.legend(fontsize=7)
        else:
            ax.text(0.5, 0.5, 'n/a', ha='center', va='center', transform=ax.transAxes, color='gray')
    fig.tight_layout()
    fig.savefig(a.out, dpi=110)
    print('saved', a.out)
    for r in a.runs + ([a.baseline] if a.baseline else []):
        d = data.get(r) or base
        row = []
        for m in ('psnr', 'ssim', 'lpips'):
            s, v = series(d, f'metrics/{a.val}/{m}')
            if v:
                row.append(f'{m} ' + ' → '.join(f'{x:.3f}' for x in v) + f' (iters {s[0]}–{s[-1]})')
        print(f'{r}:\n   ' + '\n   '.join(row))


if __name__ == '__main__':
    main()
