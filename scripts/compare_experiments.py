"""Backward-compatible wrapper for the installed ``captcha-compare`` command."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from captcha_vision.cli_scripts.compare_experiments import main


if __name__ == "__main__":
    main()
