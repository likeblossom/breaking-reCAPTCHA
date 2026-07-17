from __future__ import annotations

import torch
import torch.nn as nn

from captcha_vision.inference.predictor import Decision, Predictor, UNCERTAIN_LABEL


class ConstantModel(nn.Module):
    def __init__(self, logits: list[float]) -> None:
        super().__init__()
        self.register_buffer("logits", torch.tensor(logits, dtype=torch.float32))
        self.calls = 0

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return self.logits.expand(inputs.shape[0], -1)


def make_predictor(logits: list[float], threshold: float, tta: bool) -> Predictor:
    predictor = Predictor.__new__(Predictor)
    predictor.threshold = threshold
    predictor.tta = tta
    predictor.device = torch.device("cpu")
    predictor.model = ConstantModel(logits)
    predictor.class_names = ["Bicycle", "Car"]
    return predictor


def test_prediction_above_threshold_is_accepted() -> None:
    predictor = make_predictor([0.0, 3.0], threshold=0.8, tta=False)

    result = predictor.predict_tensor(torch.zeros(3, 8, 8), path="tile.png")

    assert result.decision is Decision.ACCEPTED
    assert result.label == "Car"
    assert result.class_idx == 1
    assert result.path == "tile.png"
    assert result.confidence > 0.9
    assert set(result.all_probs) == {"Bicycle", "Car"}


def test_prediction_below_threshold_is_routed_to_uncertain() -> None:
    predictor = make_predictor([0.0, 0.0], threshold=0.6, tta=False)

    result = predictor.predict_tensor(torch.zeros(3, 8, 8))

    assert result.decision is Decision.UNCERTAIN
    assert result.label == UNCERTAIN_LABEL
    assert result.confidence == 0.5


def test_tta_runs_original_and_horizontal_flip() -> None:
    predictor = make_predictor([2.0, 0.0], threshold=0.5, tta=True)

    predictor.predict_tensor(torch.zeros(3, 8, 8))

    assert predictor.model.calls == 2


def test_prediction_result_serializes_enum_and_probabilities() -> None:
    predictor = make_predictor([2.0, 0.0], threshold=0.5, tta=False)

    payload = predictor.predict_tensor(torch.zeros(3, 8, 8)).to_dict()

    assert payload["decision"] == "accepted"
    assert payload["prob_Bicycle"] > payload["prob_Car"]
