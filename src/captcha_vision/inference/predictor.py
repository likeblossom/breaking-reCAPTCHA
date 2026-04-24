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
from captcha_vision.data.dataset import _rgba_to_rgb, get_transforms
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

    def predict_image_with_tta(self, image_path: str | Path) -> PredictionResult:
        """Like ``predict_image`` but forces TTA on regardless of instance setting."""
        img = _rgba_to_rgb(str(image_path))
        tensor = self._transform(img).unsqueeze(0).to(self.device)
        return self._predict_tensor(tensor, path=str(image_path), tta_override=True)

    def predict_pil(self, img: "Image.Image", *, label: str = "<pil>") -> PredictionResult:
        """Predict from an in-memory PIL Image (avoids disk round-trip)."""
        from PIL import Image as _PILImage  # noqa: F811 – lazy to avoid top-level dep
        if img.mode != "RGB":
            img = img.convert("RGB")
        tensor = self._transform(img).unsqueeze(0).to(self.device)
        return self._predict_tensor(tensor, path=label)

    def score_target_image(
        self,
        image_path: str | Path,
        target: str,
        *,
        tta_override: bool | None = None,
    ) -> float:
        """
        Estimate whether ``target`` is present anywhere in an image.

        Unlike the standard classifier output, this score is target-aware:
        it keeps the maximum target probability across multiple spatial views
        instead of averaging them. That makes it more robust when a tile
        contains the target plus a distractor object.
        """
        img = _rgba_to_rgb(str(image_path))
        tensor = self._transform(img).unsqueeze(0).to(self.device)
        return self.score_target_tensor(tensor, target, tta_override=tta_override)

    def score_target_pil(
        self,
        img: "Image.Image",
        target: str,
        *,
        tta_override: bool | None = None,
    ) -> float:
        """Like ``score_target_image`` but works on an in-memory PIL image."""
        if img.mode != "RGB":
            img = img.convert("RGB")
        tensor = self._transform(img).unsqueeze(0).to(self.device)
        return self.score_target_tensor(tensor, target, tta_override=tta_override)

    def score_target_tensor(
        self,
        tensor: torch.Tensor,
        target: str,
        *,
        tta_override: bool | None = None,
    ) -> float:
        """
        Return a target-presence score in [0, 1].

        Standard softmax classification is forced to choose one class for the
        whole tile. For mixed tiles, averaging crops can dilute a true target.
        Here we treat each augmented/cropped view as an opportunity to spot the
        target and keep the maximum target probability across views.
        """
        if target not in self.class_names:
            return 0.0
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        return self._score_target_tensor(
            tensor.to(self.device), target, tta_override=tta_override
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _predict_tensor(
        self, tensor: torch.Tensor, path: str, *, tta_override: bool | None = None,
    ) -> PredictionResult:
        use_tta = self.tta if tta_override is None else tta_override

        logits = self.model(tensor)
        probs = F.softmax(logits, dim=1)

        if use_tta:
            logits_flip = self.model(TF.hflip(tensor))
            probs = (probs + F.softmax(logits_flip, dim=1)) / 2

            _, _, h, w = tensor.shape
            crop_h, crop_w = int(h * 0.85), int(w * 0.85)
            if crop_h >= 32 and crop_w >= 32:
                crops = TF.five_crop(tensor, (crop_h, crop_w))
                crop_probs = []
                for crop in crops:
                    resized = TF.resize(crop, [h, w], antialias=True)
                    crop_probs.append(F.softmax(self.model(resized), dim=1))
                avg_crop = torch.stack(crop_probs).mean(dim=0)
                probs = 0.6 * probs + 0.4 * avg_crop

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

    @torch.no_grad()
    def _score_target_tensor(
        self,
        tensor: torch.Tensor,
        target: str,
        *,
        tta_override: bool | None = None,
    ) -> float:
        use_tta = self.tta if tta_override is None else tta_override
        target_idx = self.class_names.index(target)

        views = [tensor]
        if use_tta:
            views.append(TF.hflip(tensor))

            _, _, h, w = tensor.shape
            crop_h, crop_w = int(h * 0.85), int(w * 0.85)
            if crop_h >= 32 and crop_w >= 32:
                for crop in TF.five_crop(tensor, (crop_h, crop_w)):
                    views.append(TF.resize(crop, [h, w], antialias=True))

        best = 0.0
        for view in views:
            probs = F.softmax(self.model(view), dim=1)
            best = max(best, float(probs[0, target_idx]))
        return best
