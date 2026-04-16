"""
Training script for the reCAPTCHA tile classifier.

Two-phase fine-tuning
---------------------
Phase 1  Classifier head only (backbone frozen, higher LR, fewer epochs).
Phase 2  End-to-end (all layers, lower LR, more epochs).

Each phase starts with linear LR warmup followed by cosine decay.
MixUp, focal loss, class-weighted loss, and torch.compile are all optional
flags so different training strategies can be compared without code changes.

Usage
-----
python scripts/train.py \\
    --data_dir data/dataset \\
    --epochs_phase1 5 \\
    --epochs_phase2 30 \\
    --batch_size 64 \\
    --loss focal \\
    --class_weighted_loss
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import time
from pathlib import Path
from typing import Any, TextIO

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from captcha_vision.common.device import get_device
from captcha_vision.data.dataset import build_class_weights_for_split, build_dataloaders
from captcha_vision.models.classifier import CaptchaClassifier


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """
    Focal loss for imbalanced multi-class classification.

    Down-weights easy examples (high p_t) so training focuses on hard/rare
    ones.  gamma=0 recovers standard cross-entropy.  Supports optional
    class-frequency weighting via the ``weight`` buffer.

    Lin et al., 2017 — https://arxiv.org/abs/1708.02002
    """

    def __init__(
        self,
        gamma: float = 2.0,
        weight: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        if weight is not None:
            self.register_buffer("weight", weight)
        else:
            self.weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_p = F.log_softmax(logits, dim=1)
        # Per-sample CE loss (no reduction) so we can apply the focal factor.
        ce = F.nll_loss(log_p, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return ((1.0 - pt) ** self.gamma * ce).mean()


def build_criterion(
    loss_type: str,
    class_weights: torch.Tensor | None,
    label_smoothing: float,
    focal_gamma: float,
    device: torch.device,
) -> nn.Module:
    """
    Construct the training loss.

    ``class_weights`` scales gradient magnitude by inverse class frequency,
    stacking with the ``WeightedRandomSampler`` in the DataLoader.  The two
    mechanisms are complementary: the sampler controls what the model *sees*;
    the loss weight controls how hard each misclassification is penalised.
    """
    w = class_weights.to(device) if class_weights is not None else None
    if loss_type == "focal":
        return FocalLoss(gamma=focal_gamma, weight=w).to(device)
    return nn.CrossEntropyLoss(weight=w, label_smoothing=label_smoothing)


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class PhaseConfig:
    """Hyperparameters that differ between training phases."""
    phase: int
    lr: float
    epochs: int
    warmup_epochs: int
    patience: int
    grad_clip: float
    mixup_alpha: float


@dataclasses.dataclass
class RunContext:
    """
    Objects shared across both phases.

    Separating these from ``PhaseConfig`` keeps ``_train_phase`` to three
    arguments instead of fifteen.
    """
    classifier: CaptchaClassifier      # original model — used for saving
    model: nn.Module                   # compiled or original — used for forward
    train_loader: torch.utils.data.DataLoader
    val_loader: torch.utils.data.DataLoader
    train_criterion: nn.Module
    val_criterion: nn.Module           # plain CE, no smoothing — cleaner stopping signal
    device: torch.device
    output_dir: Path
    class_names: list[str]
    csv_writer: csv.DictWriter
    csv_file: TextIO
    save_metadata: dict[str, Any]


# ---------------------------------------------------------------------------
# Core epoch runner
# ---------------------------------------------------------------------------

def _run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    desc: str,
    mixup_alpha: float = 0.0,
    grad_clip: float = 0.0,
) -> tuple[float, float]:
    """
    Run one full epoch; return (mean_loss, sample_level_accuracy).

    Accuracy is tracked as total_correct / total_samples rather than the mean
    of per-batch accuracies, so a short final batch does not skew the number.

    When optimizer is None the model runs in eval mode with no gradients.
    """
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    n_batches = 0

    with torch.set_grad_enabled(is_train):
        for images, labels in tqdm(loader, desc=desc, leave=False):
            images, labels = images.to(device), labels.to(device)

            if is_train and mixup_alpha > 0.0:
                # MixUp: blend two random samples; use two CE terms to avoid
                # constructing soft one-hot targets.
                lam = float(np.random.beta(mixup_alpha, mixup_alpha))
                idx = torch.randperm(images.size(0), device=device)
                images = lam * images + (1 - lam) * images[idx]
                logits = model(images)
                loss = (
                    lam * criterion(logits, labels)
                    + (1 - lam) * criterion(logits, labels[idx])
                )
                dominant_labels = labels if lam >= 0.5 else labels[idx]
            else:
                logits = model(images)
                loss = criterion(logits, labels)
                dominant_labels = labels

            if is_train and optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()

            total_loss += loss.item()
            total_correct += (logits.argmax(dim=1) == dominant_labels).sum().item()
            total_samples += labels.size(0)
            n_batches += 1

    return total_loss / n_batches, total_correct / total_samples


# ---------------------------------------------------------------------------
# LR scheduler
# ---------------------------------------------------------------------------

def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int,
    warmup_epochs: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Linear LR warmup, then cosine annealing to zero."""
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs - warmup_epochs, 1)
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
    )


# ---------------------------------------------------------------------------
# Phase trainer
# ---------------------------------------------------------------------------

def _train_phase(
    cfg: PhaseConfig,
    ctx: RunContext,
    best_val_acc: float,
) -> float:
    """
    Run one training phase.

    Saves the best checkpoint whenever val accuracy improves, and halts early
    if it has not improved for ``cfg.patience`` consecutive epochs.

    Returns the updated best val accuracy.
    """
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, ctx.model.parameters()),
        lr=cfg.lr,
        weight_decay=1e-4,
    )
    scheduler = _build_scheduler(optimizer, cfg.epochs, cfg.warmup_epochs)

    print(
        f"\n{'='*60}\n"
        f"  Phase {cfg.phase}"
        f"  |  {ctx.classifier.trainable_parameter_count():,} trainable params"
        f"  |  lr={cfg.lr}"
        f"  |  warmup={cfg.warmup_epochs} epochs\n"
        f"{'='*60}"
    )

    no_improve = 0

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = _run_epoch(
            ctx.model, ctx.train_loader, ctx.train_criterion,
            optimizer, ctx.device,
            desc=f"P{cfg.phase} E{epoch:02d} train",
            mixup_alpha=cfg.mixup_alpha,
            grad_clip=cfg.grad_clip,
        )
        # Val uses a plain CE criterion (no smoothing, no weighting) so the
        # stopping signal is not distorted by the training loss configuration.
        val_loss, val_acc = _run_epoch(
            ctx.model, ctx.val_loader, ctx.val_criterion,
            None, ctx.device,
            desc=f"P{cfg.phase} E{epoch:02d} val  ",
        )
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        elapsed = time.time() - t0
        is_best = val_acc > best_val_acc

        print(
            f"  Phase {cfg.phase} | Epoch {epoch:02d}/{cfg.epochs}"
            f"  train_loss={train_loss:.4f}  train_acc={train_acc:.4f}"
            f"  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
            f"  lr={current_lr:.2e}"
            f"  {'** BEST **' if is_best else f'(no improve {no_improve+1}/{cfg.patience})'}"
            f"  ({elapsed:.1f}s)"
        )

        ctx.csv_writer.writerow(
            {
                "phase": cfg.phase,
                "epoch": epoch,
                "train_loss": f"{train_loss:.6f}",
                "train_acc": f"{train_acc:.6f}",
                "val_loss": f"{val_loss:.6f}",
                "val_acc": f"{val_acc:.6f}",
                "lr": f"{current_lr:.8f}",
            }
        )
        # Flush after every row so logs survive a crash mid-training.
        ctx.csv_file.flush()

        if is_best:
            best_val_acc = val_acc
            no_improve = 0
            ckpt_path = ctx.output_dir / "best_model.pt"
            ctx.classifier.save(
                ckpt_path,
                ctx.class_names,
                best_val_acc=round(val_acc, 6),
                **ctx.save_metadata,
            )
            print(f"    Saved checkpoint → {ckpt_path}")
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                print(
                    f"  Early stopping: no improvement for {cfg.patience} epochs."
                )
                break

    return best_val_acc


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train the CAPTCHA tile classifier.")
    parser.add_argument("--data_dir", default="data/dataset")
    parser.add_argument("--output_dir", default="models")
    parser.add_argument("--log_dir", default="artifacts/logs")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs_phase1", type=int, default=5,
                        help="Head-only warmup epochs")
    parser.add_argument("--epochs_phase2", type=int, default=30,
                        help="Full fine-tuning epochs")
    parser.add_argument("--lr_phase1", type=float, default=1e-3)
    parser.add_argument("--lr_phase2", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int, default=2,
                        help="Linear LR warmup epochs at the start of each phase")
    parser.add_argument("--patience", type=int, default=8,
                        help="Early stopping patience (epochs without val improvement)")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Max gradient norm (0 = disabled)")
    parser.add_argument("--mixup_alpha", type=float, default=0.2,
                        help="MixUp Beta-distribution alpha (0 = disabled)")
    parser.add_argument("--label_smoothing", type=float, default=0.1,
                        help="Label smoothing applied to CE loss (ignored for focal)")
    parser.add_argument("--loss", choices=["ce", "focal"], default="ce",
                        help="Training loss function")
    parser.add_argument("--focal_gamma", type=float, default=2.0,
                        help="Focal loss focusing parameter γ (ignored when --loss=ce)")
    parser.add_argument("--class_weighted_loss", action="store_true",
                        help="Scale loss by inverse class frequency "
                             "(stacks with the weighted sampler)")
    parser.add_argument("--compile", action="store_true",
                        help="Apply torch.compile for faster training "
                             "(requires PyTorch 2.0+; MPS support is limited)")
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    log_dir = Path(args.log_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Device: {device}")

    print("Loading datasets...")
    train_loader, val_loader, test_loader, class_names = build_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    print(
        f"  Classes ({len(class_names)}): {class_names}\n"
        f"  Train batches: {len(train_loader)}"
        f"  |  Val: {len(val_loader)}"
        f"  |  Test: {len(test_loader)}"
    )

    class_weights = None
    if args.class_weighted_loss:
        class_weights = build_class_weights_for_split(
            args.data_dir, val_split=args.val_split, seed=args.seed
        )
        print(
            f"  Class weights: "
            + "  ".join(
                f"{name}={w:.2f}"
                for name, w in zip(class_names, class_weights.tolist())
            )
        )

    train_criterion = build_criterion(
        loss_type=args.loss,
        class_weights=class_weights,
        label_smoothing=args.label_smoothing,
        focal_gamma=args.focal_gamma,
        device=device,
    )
    # Val criterion is always plain CE so the stopping signal is consistent
    # across different --loss and --label_smoothing settings.
    val_criterion = nn.CrossEntropyLoss()

    classifier = CaptchaClassifier(
        num_classes=len(class_names), pretrained=True
    ).to(device)

    model: nn.Module = classifier
    if args.compile:
        print("Compiling model with torch.compile …")
        model = torch.compile(classifier)

    save_metadata = {
        "loss": args.loss,
        "focal_gamma": args.focal_gamma,
        "class_weighted_loss": args.class_weighted_loss,
        "label_smoothing": args.label_smoothing,
        "lr_phase1": args.lr_phase1,
        "lr_phase2": args.lr_phase2,
        "batch_size": args.batch_size,
        "epochs_phase1": args.epochs_phase1,
        "epochs_phase2": args.epochs_phase2,
        "mixup_alpha": args.mixup_alpha,
        "seed": args.seed,
    }

    log_path = log_dir / "training_log.csv"
    with open(log_path, "w", newline="") as f:
        fieldnames = [
            "phase", "epoch",
            "train_loss", "train_acc",
            "val_loss", "val_acc",
            "lr",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        ctx = RunContext(
            classifier=classifier,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            train_criterion=train_criterion,
            val_criterion=val_criterion,
            device=device,
            output_dir=output_dir,
            class_names=class_names,
            csv_writer=writer,
            csv_file=f,
            save_metadata=save_metadata,
        )

        classifier.freeze_backbone()
        best_val_acc = _train_phase(
            PhaseConfig(
                phase=1,
                lr=args.lr_phase1,
                epochs=args.epochs_phase1,
                warmup_epochs=min(args.warmup_epochs, args.epochs_phase1),
                patience=args.patience,
                grad_clip=args.grad_clip,
                mixup_alpha=args.mixup_alpha,
            ),
            ctx,
            best_val_acc=0.0,
        )

        classifier.unfreeze_backbone()
        best_val_acc = _train_phase(
            PhaseConfig(
                phase=2,
                lr=args.lr_phase2,
                epochs=args.epochs_phase2,
                warmup_epochs=min(args.warmup_epochs, args.epochs_phase2),
                patience=args.patience,
                grad_clip=args.grad_clip,
                mixup_alpha=args.mixup_alpha,
            ),
            ctx,
            best_val_acc=best_val_acc,
        )

    print(f"\nTraining complete.  Best val accuracy: {best_val_acc:.4f}")
    print(f"Checkpoint:   {output_dir / 'best_model.pt'}")
    print(f"Training log: {log_path}")


if __name__ == "__main__":
    main()
