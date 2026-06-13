# Operations

## GPU Diagnostics

Check the runtime before long training jobs:

```bash
helio-diagnose-gpu
```

For quick CPU-only validation, pass `--allow-cpu-fallback` to the training commands. For real training, keep the default strict GPU behavior so CUDA misconfiguration fails early instead of silently running a long job on CPU.

## Training Artifacts

By default, training writes under `checkpoints/`. Override this with `--save-root` when running shared experiments:

```bash
helio-train --model rcan --save-root D:\experiments\helioresolve
```

Each run writes:

- `config.json`
- `history.json`
- `summary.json`
- `best_model.pt`

These files are local artifacts and should not be committed to the public repository.

## Public Release Checklist

```bash
git status --short
git ls-files | grep -E '(^|/)(Solar Dataset|Notebooks|Models|checkpoints|outputs|runs|logs)(/|$)|\.(pt|pth|ckpt|onnx|safetensors|ipynb)$|__pycache__' && exit 1 || true
python -m pytest
```
