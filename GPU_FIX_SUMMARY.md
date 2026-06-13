# GPU Fix Summary

The training pipeline now resolves device placement centrally, reports runtime details at startup, supports mixed precision where available, and exposes `--allow-cpu-fallback` for deliberate local smoke runs.

Recommended long-run checks:

```bash
python diagnose_gpu.py
python train_sr.py --model srcnn --epochs 1 --batch-size 1 --max-train-batches 2 --max-val-batches 2
```
