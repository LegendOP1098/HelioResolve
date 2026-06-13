# GPU Debug Guide

Use `diagnose_gpu.py` to verify the runtime before long training jobs:

```bash
python diagnose_gpu.py
```

For quick CPU-only validation, pass `--allow-cpu-fallback` to the training scripts. For production training, keep the default strict GPU behavior so CUDA misconfiguration fails early instead of silently running a long job on CPU.
