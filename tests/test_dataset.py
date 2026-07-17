from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from captcha_vision.data.dataset import (
    IMAGE_SIZE,
    _rgba_to_rgb,
    build_class_weights_for_split,
    build_dataloaders,
)


def test_rgba_loader_composites_transparency_on_white(tmp_path: Path) -> None:
    path = tmp_path / "transparent.png"
    image = Image.new("RGBA", (2, 1))
    image.putdata([(255, 0, 0, 255), (0, 0, 255, 0)])
    image.save(path)

    loaded = _rgba_to_rgb(str(path))

    assert loaded.mode == "RGB"
    assert loaded.getpixel((0, 0)) == (255, 0, 0)
    assert loaded.getpixel((1, 0)) == (255, 255, 255)


def test_build_dataloaders_preserves_split_and_class_indices(tiny_dataset: Path) -> None:
    train, val, test, class_names = build_dataloaders(
        tiny_dataset,
        batch_size=4,
        val_split=0.2,
        num_workers=0,
        seed=7,
    )

    assert class_names == ["Bicycle", "Car"]
    assert len(train.dataset) == 16
    assert len(val.dataset) == 4
    assert len(test.dataset) == 4

    images, labels = next(iter(val))
    assert images.shape[1:] == (3, IMAGE_SIZE, IMAGE_SIZE)
    assert set(labels.tolist()) <= {0, 1}


def test_balanced_split_has_equal_class_weights(tiny_dataset: Path) -> None:
    weights = build_class_weights_for_split(
        tiny_dataset, val_split=0.2, seed=7, smoothing="sqrt"
    )

    torch.testing.assert_close(weights, torch.ones(2))
