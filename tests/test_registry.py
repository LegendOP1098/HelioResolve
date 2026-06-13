import torch

from solarres_sr.registry import build_model, list_available_models


def test_model_registry_contains_expected_models() -> None:
    assert {"srcnn", "rcan", "swinir", "diffusion_sr"}.issubset(set(list_available_models()))


def test_srcnn_forward_shape() -> None:
    model = build_model("srcnn", in_channels=1, out_channels=1, scale=4, capacity="tiny")
    output = model(torch.rand(1, 1, 8, 8))
    assert tuple(output.shape) == (1, 1, 32, 32)
