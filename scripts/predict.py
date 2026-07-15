"""Backward-compatible wrapper for the installed ``captcha-predict`` command."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from captcha_vision.cli_scripts.predict import main


if __name__ == "__main__":
    main()
