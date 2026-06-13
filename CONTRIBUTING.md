# Contributing

Thanks for improving HelioResolve. Keep contributions source-first and reproducible.

Before opening a pull request:

```bash
python -m pytest
python -m py_compile train_sr.py benchmark_sr_models.py finetune_sr_models.py train_psnr_max.py diagnose_gpu.py
```

Please do not commit private datasets, notebooks, trained weights, checkpoints, logs, or local environment files. Add new model code under `solarres_sr/models/` and register it in `solarres_sr/registry.py`.
