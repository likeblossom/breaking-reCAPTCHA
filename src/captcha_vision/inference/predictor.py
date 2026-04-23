"""
Inference routing for the reCAPTCHA tile classifier.

Core concept
------------
Every prediction carries a ``confidence`` score (max softmax probability).
If that score is below ``threshold`` the prediction is routed to the
``UNCERTAIN`` bucket instead of being returned as a definitive answer.
Callers handle the uncertain bucket via a fallback path (retry, manual
solver, etc.).

Typical usage
-------------
    from captcha_vision.inference.predictor import Predictor, Decision

    predictor = Predictor(
        checkpoint="models/exp_ce_weighted_sqrt/best_model.pt",
        threshold=0.75,
    )

    result = predictor.predict_image("path/to/tile.png")
    if result.decision == Decision.UNCERTAIN:
        # confidence below threshold — hand off to fallback
        ...
    else:
        print(result.label, result.confidence)
"""

from __future__ import annotations

import dataclasses
from enum import Enum
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from captcha_vision.common.device import get_device
from captcha_vision.data.dataset import IMAGE_SIZE, _rgba_to_rgb, get_transforms
from captcha_vision.models.classifier import CaptchaClassifier


UNCERTAIN_LABEL = "uncertain"


class Decision(str, Enum):
    ACCEPTED = "accepted"
    UNCERTAIN = "uncertain"


@dataclasses.dataclass
class PredictionResult:
    """
    The complete output of a single-image prediction.

    Attributes
    ----------
    path:       Source file path (empty string when predicting from a tensor).
    label:      Predicted class name, or ``UNCERTAIN_LABEL`` when rejected.
    class_idx:  Integer index of the top class (even when uncertain).
    confidence: Max softmax probability of the top class (0-1).
    decision:   ACCEPTED if confidence >= threshold, else UNCERTAIN.
    threshold:  The threshold used for routing.
    all_probs:  Full probability vector, keyed by class name.
    """

    path: str
    label: str
    class_idx: int
    confidence: float
    decision: Decision
    threshold: float
    all_probs: dict[str, float]

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "label": self.label,
            "class_idx": self.class_idx,
            "confidence": round(self.confidence, 6),
            "decision": self.decision.value,
            "threshold": self.threshold,
            **{f"prob_{k}": round(v, 6) for k, v in self.all_probs.items()},
        }


class Predictor:
    """
    Stateful inference object.  Load once, call many times.

    Parameters
    ----------
    checkpoint:
        Path to a ``best_model.pt`` saved by ``CaptchaClassifier.save()``.
    threshold:
        Confidence threshold (0-1).  Predictions below this are routed to
        the uncertain bucket.  Tune with the ``--sweep_thresholds`` flag in
        ``scripts/evaluate.py``.  Recommended default: 0.75.
    device:
        ``torch.device`` to run on.  Auto-detected if ``None``.
    tta:
        Test Time Augmentation.  Averages softmax over original + h-flip,
        giving a small accuracy boost with no retraining.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        threshold: float = 0.75,
        device: torch.device | None = None,
        tta: bool = True,
    ) -> None:
        self.threshold = threshold
        self.tta = tta
        self.device = device or get_device()
        self.model, self.class_names = CaptchaClassifier.load(
            checkpoint, device=self.device
        )
        self.model.eval()
        self._transform = get_transforms(train=False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_image(self, image_path: str | Path) -> PredictionResult:
        """Predict a single image file."""
        img = _rgba_to_rgb(str(image_path))
        tensor = self._transform(img).unsqueeze(0).to(self.device)
        return self._predict_tensor(tensor, path=str(image_path))

    def predict_batch(
        self, image_paths: list[str | Path]
    ) -> list[PredictionResult]:
        """Predict a list of image files, one at a time (preserves order)."""
        return [self.predict_image(p) for p in image_paths]

    def predict_tensor(
        self,
        tensor: torch.Tensor,
        path: str = "",
    ) -> PredictionResult:
        """
        Predict a pre-processed image tensor.

        The tensor must already be normalised (i.e. output of
        ``get_transforms(train=False)``).  Shape: (C, H, W) or (1, C, H, W).
        """
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        return self._predict_tensor(tensor.to(self.device), path=path)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _predict_tensor(
        self, tensor: torch.Tensor, path: str
    ) -> PredictionResult:
        logits = self.model(tensor)
        probs = F.softmax(logits, dim=1)

        if self.tta:
            logits_flip = self.model(TF.hflip(tensor))
            probs = (probs + F.softmax(logits_flip, dim=1)) / 2

        probs_vec = probs[0]
        confidence = float(probs_vec.max())
        class_idx = int(probs_vec.argmax())
        top_label = self.class_names[class_idx]

        decision = (
            Decision.ACCEPTED if confidence >= self.threshold else Decision.UNCERTAIN
        )
        label = top_label if decision == Decision.ACCEPTED else UNCERTAIN_LABEL

        all_probs = {
            name: float(probs_vec[i])
            for i, name in enumerate(self.class_names)
        }

        return PredictionResult(
            path=path,
            label=label,
            class_idx=class_idx,
            confidence=confidence,
            decision=decision,
            threshold=self.threshold,
            all_probs=all_probs,
        )
