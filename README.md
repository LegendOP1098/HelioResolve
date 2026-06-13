# SolarRes SR

SolarRes SR is a PyTorch training suite for solar image super-resolution. It includes a reusable package, CLI scripts, and model registry for comparing SRCNN, RLFB+ESA, EDSR, RCAN, SwinIR, SRGAN, ESRGAN, and conditional diffusion SR on the same low-resolution/high-resolution solar dataset layout.

## What Is Included

- Reusable package under `solarres_sr/`
- Training CLI: `train_sr.py`
- Benchmark CLI: `benchmark_sr_models.py`
- Fine-tuning/search CLI: `finetune_sr_models.py`
- PSNR-focused training workflow: `train_psnr_max.py`
- Lightweight tests and CI configuration
- Public-safe ignore rules for datasets, notebooks, checkpoints, weights, logs, and secrets

## Repository Layout

```text
solarres_sr/              Core package: data, losses, metrics, registry, training, models
tests/                    Fast smoke tests that do not require private data
docs/                     Data layout, repo hygiene, and operational notes
train_sr.py               Train a single model
benchmark_sr_models.py    Compare candidate models
finetune_sr_models.py     Quick search plus optional final training
train_psnr_max.py         Long-running PSNR-optimized training workflow
```

Local research assets are intentionally ignored by Git: `Solar Dataset/`, `Notebooks/`, `Models/`, `checkpoints/`, `outputs/`, `runs/`, and model weight files such as `.pt` and `.pth`.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

On Linux/macOS, replace the activation command with `source .venv/bin/activate`.

## Dataset Layout

Keep the dataset outside Git. The default resolver looks for this structure under the project root:

```text
Solar Dataset/
  Solar Dataset/
    training/
      low_res/
      high_res/
    validation/
      low_res/
      high_res/
```

The split names `train`/`val` and `training`/`validation` are both supported. You can pass either the project root or the actual dataset root to `--dataset-root`.

## Train

```bash
python train_sr.py --model diffusion_sr --epochs 50 --batch-size 4
python train_sr.py --model rcan --epochs 50 --batch-size 4
python train_sr.py --model edsr --epochs 50 --batch-size 4
```

Use `--allow-cpu-fallback` for local smoke runs on machines without CUDA:

```bash
python train_sr.py --model srcnn --epochs 1 --batch-size 1 --allow-cpu-fallback --max-train-batches 2 --max-val-batches 2
```

## Benchmark

```bash
python benchmark_sr_models.py --models diffusion_sr rcan edsr --epochs 40 --batch-size 4
```

## Outputs

Training writes run artifacts under `checkpoints/` by default, or under `--save-root` when provided. These artifacts are intentionally ignored by Git:

- `config.json`
- `history.json`
- `summary.json`
- `best_model.pt`

## Privacy

Do not commit private datasets, notebooks with rendered outputs, trained weights, checkpoints, logs, `.env` files, API keys, or cloud credentials. See `docs/REPOSITORY_HYGIENE.md` for the repo hygiene checklist.
