"""
EfficientNet-B2 classifier for reCAPTCHA tile classification.

Two-phase training plan
-----------------------
Phase 1  Freeze the backbone and train only the new classifier head.
         This helps the new head learn first before updating the pretrained
         weights.

Phase 2  Unfreeze the full model and train end-to-end with a smaller learning
         rate so pretrained ImageNet features adjust smoothly to CAPTCHA tiles.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import EfficientNet_B2_Weights, efficientnet_b2


class CaptchaClassifier(nn.Module):
    """EfficientNet-B2 with a replaced classification head."""

    def __init__(self, num_classes: int, pretrained: bool = True) -> None:
        super().__init__()
        weights = EfficientNet_B2_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = efficientnet_b2(weights=weights)

        # Replace the 1000-class ImageNet head with our num_classes head.
        in_features: int = self.backbone.classifier[1].in_features  # 1408 for B2
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    # ------------------------------------------------------------------
    # Phase helpers
    # ------------------------------------------------------------------

    def freeze_backbone(self) -> None:
        """Freeze all layers except the classifier head (Phase 1)."""
        for name, param in self.backbone.named_parameters():
            param.requires_grad = "classifier" in name

    def unfreeze_backbone(self) -> None:
        """Unfreeze all parameters for end-to-end fine-tuning (Phase 2)."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path, class_names: list[str], **metadata) -> None:
        """
        Persist the model.  Any keyword arguments are stored alongside the
        weights so a checkpoint is self-describing (loss type, LRs, etc.).
        """
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "num_classes": len(class_names),
                "class_names": class_names,
                "arch": "efficientnet_b2",
                **metadata,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device: torch.device | str = "cpu") -> tuple["CaptchaClassifier", list[str]]:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model = cls(num_classes=checkpoint["num_classes"], pretrained=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        return model, checkpoint["class_names"]
