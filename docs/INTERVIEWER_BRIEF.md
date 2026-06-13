# Interviewer Brief

HelioResolve is a solar image super-resolution project designed to show both machine-learning experimentation and practical software engineering.

## Project Pitch

Low-resolution solar observations can lose fine magnetic and structural detail. HelioResolve builds a reproducible PyTorch pipeline to train, benchmark, and compare multiple super-resolution model families on aligned low-resolution/high-resolution solar image pairs.

## Technical Scope

- Dataset validation with strict LR/HR filename pairing to avoid silent supervision errors.
- Solar-specific preprocessing options, including multi-channel feature extraction.
- Unified model registry for baseline CNNs, the custom RESM architecture, residual attention models, SwinIR, GANs, and conditional diffusion SR.
- Shared training abstractions for optimizers, schedulers, AMP, EMA, checkpointing, validation, early stopping, and metrics.
- Benchmark workflows that rank candidates using PSNR, SSIM, RMSE, correlation, bicubic gap, and a composite score.
- Public-safe release process that keeps private data, notebooks, checkpoints, and model weights out of GitHub.

## Strong Files To Discuss

- `solarres_sr/training.py`: central orchestration layer for training and benchmarking.
- `solarres_sr/registry.py`: model selection, capacity presets, and family-aware construction.
- `solarres_sr/models/resm.py`: custom Residual Edge-aware Solar Module Network.
- `solarres_sr/data.py`: dataset resolution, pairing guarantees, preprocessing, augmentation, and crop alignment.
- `solarres_sr/models/swinir.py`: packaged SwinIR transformer implementation.
- `train_psnr_max.py`: long-running PSNR-oriented experiment workflow.

## Talking Points

- Why a common benchmark harness matters when comparing architectures.
- Why RESM uses residual learning, channel attention, and edge-aware gating for solar detail recovery.
- How LR/HR pairing bugs can invalidate super-resolution training and how the repo prevents that.
- Why diffusion SR is useful for detail recovery but expensive to evaluate.
- How public release hygiene was handled without exposing private data or trained weights.
- How the code can be extended with a new model through the registry.

## Next Improvements

- Add public sample images or synthetic fixtures for a visual demo.
- Add model cards for the strongest trained candidates.
- Add a small inference CLI once public weights are available.
- Add experiment reports with anonymized metrics and plots.
