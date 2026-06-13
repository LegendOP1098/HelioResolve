from __future__ import annotations

import platform

import torch


def main() -> None:
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA build: {torch.version.cuda}")

    if not torch.cuda.is_available():
        print("No CUDA device detected. Use --allow-cpu-fallback only for smoke tests.")
        return

    device_count = torch.cuda.device_count()
    print(f"CUDA devices: {device_count}")
    for index in range(device_count):
        props = torch.cuda.get_device_properties(index)
        total_gb = props.total_memory / (1024**3)
        print(f"[{index}] {props.name} | capability {props.major}.{props.minor} | {total_gb:.1f} GB")

    tensor = torch.ones((1024, 1024), device="cuda")
    result = tensor @ tensor
    torch.cuda.synchronize()
    print(f"CUDA matmul smoke test: ok ({float(result[0, 0]):.1f})")


if __name__ == "__main__":
    main()
