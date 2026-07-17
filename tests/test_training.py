from __future__ import annotations

import torch
import torch.nn.functional as F

from captcha_vision.training.train import FocalLoss, build_criterion


def test_focal_loss_gamma_zero_matches_cross_entropy() -> None:
    logits = torch.tensor([[2.0, -1.0], [-0.5, 1.5]], requires_grad=True)
    targets = torch.tensor([0, 1])

    actual = FocalLoss(gamma=0.0)(logits, targets)
    expected = F.cross_entropy(logits, targets)

    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_build_criterion_moves_class_weights_to_requested_device() -> None:
    weights = torch.tensor([0.5, 1.5])

    criterion = build_criterion("focal", weights, 0.1, 2.0, torch.device("cpu"))

    assert isinstance(criterion, FocalLoss)
    torch.testing.assert_close(criterion.weight, weights)
