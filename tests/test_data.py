from pathlib import Path

import numpy as np
from PIL import Image

from solarres_sr.data import SolarSRDataset, find_dataset_root, resolve_split_dirs


def _write_gray_image(path: Path, size: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = size
    image = np.arange(width * height, dtype=np.uint8).reshape(height, width)
    Image.fromarray(image, mode="L").save(path)


def _make_dataset(root: Path) -> Path:
    dataset_root = root / "Solar Dataset" / "Solar Dataset"
    for split in ("training", "validation"):
        _write_gray_image(dataset_root / split / "low_res" / "sample_001.png", (8, 8))
        _write_gray_image(dataset_root / split / "high_res" / "sample_001.png", (32, 32))
    return dataset_root


def test_dataset_root_can_be_project_root_or_actual_dataset_root(tmp_path: Path) -> None:
    dataset_root = _make_dataset(tmp_path)

    assert find_dataset_root(tmp_path) == dataset_root
    assert find_dataset_root(dataset_root) == dataset_root

    dirs = resolve_split_dirs(dataset_root)
    assert dirs["train_lr"] == dataset_root / "training" / "low_res"
    assert dirs["val_hr"] == dataset_root / "validation" / "high_res"


def test_solar_sr_dataset_loads_grayscale_pair(tmp_path: Path) -> None:
    dataset_root = _make_dataset(tmp_path)
    dirs = resolve_split_dirs(dataset_root)

    dataset = SolarSRDataset(dirs["train_lr"], dirs["train_hr"], input_mode="grayscale")

    lr, hr = dataset[0]
    assert len(dataset) == 1
    assert tuple(lr.shape) == (1, 8, 8)
    assert tuple(hr.shape) == (1, 32, 32)
