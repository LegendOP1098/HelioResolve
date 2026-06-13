from __future__ import annotations

import pytest
import torch

from solarres_sr.registry import build_model


def test_gpu_forward_smoke() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available on this machine.")

    model = build_model("srcnn", in_channels=1, capacity="tiny").cuda()
    sample = torch.rand(1, 1, 8, 8, device="cuda")

    with torch.no_grad():
        output = model(sample)

    assert tuple(output.shape) == (1, 1, 32, 32)
