# Breaking-reCAPTCHA (PyTorch)

PyTorch pipeline for reCAPTCHA tile classification.

## Folders

- `src/captcha_vision/` — package code (`data`, `models`, `training`, `common`)
- `data/dataset/` — train/test images
- `models/` — checkpoints
- `artifacts/logs/` — logs and evaluation outputs
- `scripts/` — runnable entry scripts
- `configs/` — default config files

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
```

For development, use `python -m pip install -e .` to make source changes
available without reinstalling.

Install the test dependencies and run the unit suite with:

```bash
python -m pip install -e '.[dev]'
pytest
```

Playwright also needs a browser for the optional browser workflow:

```bash
playwright install chromium
```

## Commands

```bash
captcha-train --data_dir data/dataset --output_dir models
captcha-evaluate --checkpoint models/best_model.pt --data_dir data/dataset
captcha-predict --input path/to/tile.png --checkpoint models/best_model.pt
captcha-compare artifacts/logs/baseline artifacts/logs/experiment
```

Run `captcha-solve --help` for the optional authorized browser-testing workflow.
The legacy `python scripts/<command>.py` forms remain available when running
from a source checkout.

To see all options for any command:

```bash
captcha-train --help
captcha-evaluate --help
captcha-predict --help
captcha-solve --help
captcha-compare --help
```
