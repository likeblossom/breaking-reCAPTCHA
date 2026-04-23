"""
Compare per-class metrics across experiments.

Reads ``metrics.json`` files produced by ``scripts/evaluate.py`` and prints
a focused table of precision / recall / F1 for the hard classes plus overall
accuracy.  Differences are shown relative to the first (baseline) experiment.

Usage
-----
python scripts/compare_experiments.py \\
    artifacts/logs/exp_ce_baseline_eval \\
    artifacts/logs/exp_focal_eval \\
    artifacts/logs/exp_ce_weighted_eval \\
    --focus "Bridge,Mountain,Other,Traffic Light"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# ANSI colours (fall back gracefully on terminals that strip them).
_GREEN = "\033[32m"
_RED = "\033[31m"
_RESET = "\033[0m"
_BOLD = "\033[1m"


def _colour(value: float, better: bool) -> str:
    """Wrap a numeric delta string in green (improvement) or red (regression)."""
    if abs(value) < 0.001:
        return f"{value:+.4f}"
    colour = _GREEN if better else _RED
    return f"{colour}{value:+.4f}{_RESET}"


def _load(path: Path) -> dict:
    metrics_file = path / "metrics.json"
    if not metrics_file.exists():
        raise FileNotFoundError(
            f"No metrics.json found in {path}. "
            "Run scripts/evaluate.py --output_dir {path} first."
        )
    return json.loads(metrics_file.read_text())


def _label(path: Path) -> str:
    """Use the directory name as a short experiment label."""
    return path.name


def _macro_metric_key(metric: str) -> str:
    """Map CLI metric name to sklearn's macro-average key names."""
    return "f1-score" if metric == "f1" else metric


def _print_table(
    exp_dirs: list[Path],
    focus_classes: list[str],
    metric: str = "f1",
) -> None:
    all_metrics = [_load(d) for d in exp_dirs]
    labels = [_label(d) for d in exp_dirs]

    # --- Overall accuracy ---
    print(f"\n{_BOLD}Overall accuracy{_RESET}")
    header = f"  {'Experiment':<42}  {'accuracy':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    baseline_acc = all_metrics[0]["overall_accuracy"]
    for label, m in zip(labels, all_metrics):
        acc = m["overall_accuracy"]
        delta = acc - baseline_acc
        delta_str = "" if delta == 0 else f"  {_colour(delta, delta > 0)}"
        print(f"  {label:<42}  {acc:>10.4f}{delta_str}")

    # --- Per-class table ---
    metric_label = {"f1": "F1", "precision": "Precision", "recall": "Recall"}[metric]
    print(f"\n{_BOLD}Per-class {metric_label}{_RESET}  (↑ is better; delta vs first experiment)")
    col_w = 10
    header_cols = f"  {'Class':<18}" + "".join(f"  {l[:col_w]:>{col_w}}" for l in labels)
    print(header_cols)
    print("  " + "-" * (len(header_cols) - 2))

    for cls in focus_classes:
        row = f"  {cls:<18}"
        baseline_val = all_metrics[0]["per_class"].get(cls, {}).get(metric, float("nan"))
        for i, (label, m) in enumerate(zip(labels, all_metrics)):
            val = m["per_class"].get(cls, {}).get(metric, float("nan"))
            cell = f"{val:.4f}"
            if i > 0 and not (baseline_val != baseline_val):  # not nan
                delta = val - baseline_val
                cell = f"{val:.4f} {_colour(delta, delta > 0)}"
            row += f"  {cell:>{col_w + 14 if i > 0 else col_w}}"
        print(row)

    # --- Macro avg ---
    print()
    row = f"  {'macro avg':<18}"
    macro_key = _macro_metric_key(metric)
    baseline_macro = all_metrics[0]["macro_avg"].get(macro_key, float("nan"))
    for i, (label, m) in enumerate(zip(labels, all_metrics)):
        val = m["macro_avg"].get(macro_key, float("nan"))
        cell = f"{val:.4f}"
        if i > 0:
            delta = val - baseline_macro
            cell = f"{val:.4f} {_colour(delta, delta > 0)}"
        row += f"  {cell:>{col_w + 14 if i > 0 else col_w}}"
    print(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare per-class eval metrics across experiments."
    )
    parser.add_argument(
        "eval_dirs",
        nargs="+",
        type=Path,
        help="Directories containing metrics.json (first = baseline).",
    )
    parser.add_argument(
        "--focus",
        default="Bridge,Mountain,Other,Traffic Light",
        help="Comma-separated class names to highlight (default: hard classes).",
    )
    parser.add_argument(
        "--metric",
        choices=["f1", "precision", "recall"],
        default="f1",
        help="Which metric to compare (default: f1).",
    )
    args = parser.parse_args()

    focus_classes = [c.strip() for c in args.focus.split(",")]
    print(f"Comparing {len(args.eval_dirs)} experiments")
    print(f"Focus classes: {focus_classes}  |  Metric: {args.metric}")

    _print_table(args.eval_dirs, focus_classes, metric=args.metric)
    print()


if __name__ == "__main__":
    main()
