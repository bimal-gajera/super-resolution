# Downstream tasks

Studies of whether Sentinel-2 super-resolved with our models helps a downstream task. Each study is self-contained
(its own scripts, Slurm files and environments) and does not import the `srbench` package (only
`scripts/registration.py`, in the superseded first check). Data, runs, results, logs, weights, generated configs,
third-party code, notes and the Slurm scripts (cluster-specific) are git-ignored.

| folder | task | downstream model |
|---|---|---|
| `fields/` | field-boundary delineation on Fields of The World (FTW) labels, following Fields of the Planet (FTP, arXiv 2607.04449) | the paper's U-Net + EfficientNet-B3 / B7, trained and scored with its own code (github.com/taylor-geospatial/fields-of-the-planet, commit 56893bc, cloned into `fields/third_party/`) |

## fields/

Step-by-step commands: [`FIELDS_PIPELINE.md`](FIELDS_PIPELINE.md).

**Setup.** The Sentinel-2 and PlanetScope rows of the comparison are taken from the FTP paper; only the SR rows are
run here. Data: the FTP paper's chip sets (FTP's `index.parquet`, FTW's splits), with FTW's own chips. FTW's chips are
used as delivered (treated as 10 m / px, as the FTW paper states; no resampling). Their 12-band versions are made with
FTW's own chip-creation procedure (github.com/fieldsoftheworld/ftw-dataset-creation, commit 812de88, cloned into
`fields/third_party/`), checked against the published 4-band chips.

Environments (under `super-res/envs/`): `ftp` (the paper's code: a venv on top of `env_sr` with
`pip install -e third_party/fields-of-the-planet` and `torchgeo==0.9.0` as its `uv.lock` pins; also the 12-band chip
scripts), `ldm` (the 3 m LDM: Python 3.10, torch 2.4.1, pytorch-lightning 1.9.5, taming-transformers 3ba01b2, as its
`README_BENCH.md` lists), `env_sr` (PixelDiT sampling, chip selection).

Row "3" (3 m SR model -> the paper's model):

| step | script |
|---|---|
| chip lists (the paper's train / val / test) | `scripts/make_chip_lists.py` (FTP's `index.parquet` + FTW's `chips_<country>.parquet`) |
| 12-band FTW chips, FTW's procedure, compared with the published chips | `scripts/ftw12_chips.py`, `slurm/ftw12_download.sbatch` |
| second pass: FTW's original scene where the catalog changed since | `scripts/ftw12_repair.py`, `slurm/ftw12_repair.sbatch` |
| final lists: chips whose two dates are FTW's exact chips | `scripts/finalize_ftw12.py` |
| FTW polygons (labels) | `scripts/download_polygons.py` |
| 3 m SR with the user's `predict_ldm.py` (`--stride 36`), then NIR (user's `stack_nir.py`) + FTP-style labels + the paper's data layout | `slurm/ldm3m_sr.sbatch`, `scripts/make_sr3m_layout.py` |
| training config = the paper's, with data / countries / epochs / logging changed | `scripts/make_config.py` |
| train, then score with the paper's evaluation + panels (Sentinel-2 / SR / labels / predictions) | `slurm/ldm3m_train.sbatch`, `scripts/evaluate_ftp.py` |
| which Maxar training pairs lie on FTW chips (none) | `scripts/maxar_overlap.py` |

Superseded first check and short run (October 2026; kept for the record). They resampled FTW's chips onto a 10 m UTM
grid before SR, which was dropped on 2026-10-25: `scripts/ftw_geo.py` (`read_lr`), `scripts/select_chips.py`,
`scripts/sr_pixeldit.py`, `scripts/mosaic.py`, `scripts/first_check.py`, `scripts/outline_offset.py`,
`scripts/make_ftp_layout.py`, `slurm/first_check.sbatch`, `slurm/ftp_train_eval.sbatch`.
