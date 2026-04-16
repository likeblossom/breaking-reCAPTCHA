"""
Evaluation script for the reCAPTCHA tile classifier.

Loads the best checkpoint and runs it on the test set. It prints
precision / recall / F1 for each class and saves a confusion matrix.

Test Time Augmentation (TTA) is on by default. Each image is evaluated
twice (original + horizontally flipped), then the softmax scores are averaged
before choosing the final class. This often gives a small accuracy boost
without retraining.

Usage
-----
python -m captcha_vision.training.evaluate \\
    --checkpoint models/best_model.pt \\
    --data_dir data/dataset \\
    --output_dir logs \\
    --tta          # enabled by default, pass --no_tta to disable
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from captcha_vision.common.device import get_device
from captcha_vision.data.dataset import build_test_loader
from captcha_vision.models.classifier import CaptchaClassifier


def _collect_predictions(
    model: CaptchaClassifier,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    tta: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run the model over the loader and return (y_true, y_pred).

    If TTA is enabled, each image is run twice: once as-is and once
    horizontally flipped. The two probability outputs are averaged before
    selecting the predicted class. This does not need retraining, but it
    doubles inference passes per image.
    """
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            images = images.to(device)

            logits = model(images)
            probs = F.softmax(logits, dim=1)

            if tta:
                logits_flip = model(TF.hflip(images))
                probs = (probs + F.softmax(logits_flip, dim=1)) / 2

            preds = probs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_preds)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    output_path: Path,
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    # Normalize each row so values show per-class prediction fractions.
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(13, 11))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        ax=ax,
        vmin=0,
        vmax=1,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Confusion Matrix (row-normalised)", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved → {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the reCAPTCHA tile classifier.")
    parser.add_argument("--checkpoint", default="models/best_model.pt")
    parser.add_argument("--data_dir", default="data/dataset")
    parser.add_argument("--output_dir", default="artifacts/logs")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--no_tta", action="store_true", help="Disable Test Time Augmentation")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Device: {device}")

    print(f"Loading checkpoint: {args.checkpoint}")
    model, class_names = CaptchaClassifier.load(args.checkpoint, device=device)
    print(f"  Classes: {class_names}")

    # build_test_loader is faster here than full build_dataloaders(), because
    # we only need test data and can skip train/val split setup.
    test_loader = build_test_loader(
        args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers
    )

    tta = not args.no_tta
    print(f"TTA: {'enabled' if tta else 'disabled'}")

    y_true, y_pred = _collect_predictions(model, test_loader, device, tta=tta)

    print("\n" + "=" * 60)
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    print("=" * 60)

    overall_acc = (y_true == y_pred).mean()
    print(f"\nOverall test accuracy: {overall_acc:.4f}")

    plot_confusion_matrix(
        y_true, y_pred, class_names, output_dir / "confusion_matrix.png"
    )


if __name__ == "__main__":
    main()
