"""
Inference routing script for reCAPTCHA tile classification.

Accepts a single image, a directory, or a glob pattern.
Predictions below ``--threshold`` are routed to the ``uncertain`` bucket.

Output modes
------------
--output_format console  Human-readable table printed to stdout (default).
--output_format json     One JSON object per line (newline-delimited JSON).
--output_format csv      CSV file written to ``--output_dir/predictions.csv``.

Usage
-----
# Single image
python scripts/predict.py --input path/to/tile.png

# Whole directory
python scripts/predict.py --input data/dataset/test/Car --output_format csv

# Custom threshold
python scripts/predict.py \\
    --input data/dataset/test \\
    --checkpoint models/exp_ce_weighted_sqrt/best_model.pt \\
    --threshold 0.75 \\
    --output_format csv \\
    --output_dir artifacts/predictions
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from captcha_vision.inference.predictor import Decision, Predictor

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


def _collect_image_paths(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(
            p for p in input_path.rglob("*") if p.suffix.lower() in _IMAGE_SUFFIXES
        )
    # Treat as a glob relative to cwd.
    paths = sorted(
        p for p in Path().glob(str(input_path))
        if p.suffix.lower() in _IMAGE_SUFFIXES
    )
    if not paths:
        print(f"[error] No images found at: {input_path}", file=sys.stderr)
        sys.exit(1)
    return paths


def _print_console(predictor: Predictor, image_paths: list[Path]) -> None:
    """Pretty-print results with an accepted/uncertain summary at the end."""
    col = {"path": 40, "label": 14, "conf": 8, "decision": 10}
    header = (
        f"{'path':<{col['path']}}  {'label':<{col['label']}}"
        f"  {'conf':>{col['conf']}}  {'decision':<{col['decision']}}"
    )
    print(header)
    print("-" * len(header))

    accepted_count = 0
    for path in image_paths:
        result = predictor.predict_image(path)
        short = str(path)
        if len(short) > col["path"]:
            short = "…" + short[-(col["path"] - 1):]
        decision_str = (
            "\033[32mACCEPTED\033[0m"
            if result.decision == Decision.ACCEPTED
            else "\033[33mUNCERTAIN\033[0m"
        )
        print(
            f"{short:<{col['path']}}  {result.label:<{col['label']}}"
            f"  {result.confidence:>{col['conf']}.4f}  {decision_str}"
        )
        if result.decision == Decision.ACCEPTED:
            accepted_count += 1

    total = len(image_paths)
    print("-" * len(header))
    print(
        f"Total: {total}  |  Accepted: {accepted_count} ({accepted_count/total:.1%})"
        f"  |  Uncertain: {total - accepted_count} ({(total - accepted_count)/total:.1%})"
    )


def _output_json(predictor: Predictor, image_paths: list[Path]) -> None:
    """Write one JSON object per line to stdout (newline-delimited JSON)."""
    for path in image_paths:
        result = predictor.predict_image(path)
        print(json.dumps(result.to_dict()))


def _output_csv(
    predictor: Predictor, image_paths: list[Path], output_dir: Path
) -> None:
    """Write all predictions to a CSV file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "predictions.csv"

    results = [predictor.predict_image(p) for p in image_paths]
    if not results:
        print("[warn] No images to predict.", file=sys.stderr)
        return

    fieldnames = list(results[0].to_dict().keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r.to_dict())

    accepted = sum(1 for r in results if r.decision == Decision.ACCEPTED)
    total = len(results)
    print(f"Predictions written → {out_path}")
    print(
        f"Total: {total}  |  Accepted: {accepted} ({accepted/total:.1%})"
        f"  |  Uncertain: {total - accepted} ({(total - accepted)/total:.1%})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run inference routing on CAPTCHA tile images."
    )
    parser.add_argument(
        "--input", required=True, type=Path,
        help="Image file, directory, or glob pattern.",
    )
    parser.add_argument(
        "--checkpoint",
        default="models/exp_ce_weighted_sqrt/best_model.pt",
        help="Path to model checkpoint.",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.75,
        help="Confidence threshold for routing (default: 0.75).",
    )
    parser.add_argument(
        "--output_format", choices=["console", "json", "csv"], default="console",
        help="Output format (default: console).",
    )
    parser.add_argument(
        "--output_dir", type=Path, default=Path("artifacts/predictions"),
        help="Output directory for CSV mode.",
    )
    parser.add_argument(
        "--no_tta", action="store_true",
        help="Disable Test Time Augmentation.",
    )
    args = parser.parse_args()

    predictor = Predictor(
        checkpoint=args.checkpoint,
        threshold=args.threshold,
        tta=not args.no_tta,
    )
    print(
        f"Loaded checkpoint: {args.checkpoint}"
        f"  |  threshold={args.threshold}"
        f"  |  TTA={'on' if not args.no_tta else 'off'}"
        f"  |  device={predictor.device}",
        file=sys.stderr,
    )

    image_paths = _collect_image_paths(args.input)
    print(f"Images found: {len(image_paths)}", file=sys.stderr)

    if args.output_format == "json":
        _output_json(predictor, image_paths)
    elif args.output_format == "csv":
        _output_csv(predictor, image_paths, args.output_dir)
    else:
        _print_console(predictor, image_paths)


if __name__ == "__main__":
    main()
