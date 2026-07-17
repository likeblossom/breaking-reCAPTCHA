from __future__ import annotations

from pathlib import Path

import torch

from captcha_vision.models.classifier import CaptchaClassifier


def test_checkpoint_save_and_load_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "model.pt"
    original = CaptchaClassifier(num_classes=2, pretrained=False)
    original.save(path, ["Bicycle", "Car"], experiment="unit-test")

    restored, class_names = CaptchaClassifier.load(path, device="cpu")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    assert class_names == ["Bicycle", "Car"]
    assert checkpoint["experiment"] == "unit-test"
    assert checkpoint["arch"] == "efficientnet_b2"
    original_state = original.state_dict()
    restored_state = restored.state_dict()
    assert original_state.keys() == restored_state.keys()
    for name in original_state:
        torch.testing.assert_close(original_state[name], restored_state[name])
