from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture
def tiny_dataset(tmp_path: Path) -> Path:
    """Create a balanced ImageFolder-style train/test dataset."""
    root = tmp_path / "dataset"
    colours = {"Bicycle": (20, 80, 140), "Car": (180, 40, 20)}

    for split, count in (("train", 10), ("test", 2)):
        for class_name, colour in colours.items():
            class_dir = root / split / class_name
            class_dir.mkdir(parents=True)
            for index in range(count):
                Image.new("RGB", (24, 18), colour).save(class_dir / f"{index}.png")

    return root
