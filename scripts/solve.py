"""
End-to-end reCAPTCHA solver using Playwright + the trained tile classifier.

Usage
-----
Basic (headed browser, demo page, default checkpoint):

    source .venv/bin/activate
    python scripts/solve.py

Custom checkpoint / threshold:

    python scripts/solve.py \\
        --checkpoint models/recaptcha_tiles_run1/best_model.pt \\
        --threshold 0.80

Headless mode (less detectable on many sites):

    python scripts/solve.py --headless

Persist browser cookies/history across runs (strongly recommended,
Google's risk engine scores you much lower when it sees real browsing history):

    python scripts/solve.py --user_data_dir ~/.chrome_captcha_profile
"""
from __future__ import annotations

import argparse
import logging
import sys

from playwright.sync_api import sync_playwright

from captcha_vision.inference.predictor import Predictor
from captcha_vision.solver.browser import launch
from captcha_vision.solver.session import SolverSession

_DEMO_URL = "https://www.google.com/recaptcha/api2/demo"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Solve reCAPTCHAv2 challenges end-to-end.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--url",
        default=_DEMO_URL,
        help="URL that contains the reCAPTCHA to solve.",
    )
    p.add_argument(
        "--checkpoint",
        default="models/recaptcha_tiles_run1/best_model.pt",
        help="Path to the trained model checkpoint (.pt file).",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.60,
        help=(
            "Confidence threshold for 3x3 tile classification (default: 0.60).  "
            "The reference project uses 0.20; anything above 0.70 will skip "
            "too many valid tiles for the solver to pass."
        ),
    )
    p.add_argument(
        "--top_n",
        type=int,
        default=3,
        help=(
            "For 4x4 grids: always click the N tiles with the highest "
            "target-class probability (top-N strategy from the reference "
            "paper).  Guarantees we never submit an empty grid.  "
            "Default 3 matches aplesner/Breaking-reCAPTCHAv2."
        ),
    )
    p.add_argument(
        "--threshold_4x4",
        type=float,
        default=0.35,
        help=(
            "Confidence threshold for 4x4 tile classification (default: 0.35).  "
            "Partial crops of a large image naturally score lower."
        ),
    )
    p.add_argument(
        "--headless",
        action="store_true",
        help="Run Chromium without a visible window.",
    )
    p.add_argument(
        "--user_data_dir",
        default=None,
        metavar="PATH",
        help=(
            "Persistent Chrome profile directory.  Reusing a real browser "
            "profile (cookies, history) is the single most effective way to "
            "reduce the number of image challenges Google presents."
        ),
    )
    p.add_argument(
        "--max_attempts",
        type=int,
        default=5,
        help="Maximum number of page-reload attempts before giving up.",
    )
    p.add_argument(
        "--output_dir",
        default="artifacts/solver_logs",
        metavar="DIR",
        help="Directory for tile screenshots and the session CSV log.",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-tile classification output.",
    )
    p.add_argument(
        "--save_tiles",
        action="store_true",
        help="Save every tile screenshot to disk (slower, useful for debugging).",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    print(f"Loading checkpoint : {args.checkpoint}")
    predictor = Predictor(
        checkpoint=args.checkpoint,
        threshold=args.threshold,
        tta=False,  # TTA doubles inference time per tile — too slow for solving
    )
    print(
        f"  Classes      : {predictor.class_names}\n"
        f"  Threshold 3x3: {args.threshold}"
        f"  |  Threshold 4x4: {args.threshold_4x4}"
        f"  |  TTA: off (speed)"
        f"  |  Device: {predictor.device}\n"
    )

    session = SolverSession(
        predictor,
        save_dir=args.output_dir,
        verbose=not args.quiet,
        threshold_4x4=args.threshold_4x4,
        save_tiles=args.save_tiles,
        top_n=args.top_n,
    )

    print(f"Target URL : {args.url}")
    print(f"Tile screenshots + session log → {args.output_dir}\n")

    with sync_playwright() as pw:
        ctx, page = launch(
            pw,
            headless=args.headless,
            user_data_dir=args.user_data_dir,
        )
        try:
            solved = session.run(
                page,
                args.url,
                max_attempts=args.max_attempts,
            )
        finally:
            ctx.close()

    if solved:
        print("\nreCAPTCHA solved successfully.")
        sys.exit(0)
    else:
        print("\nFailed to solve reCAPTCHA within the attempt limit.")
        sys.exit(1)


if __name__ == "__main__":
    main()
