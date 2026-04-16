# Breaking-reCAPTCHA (PyTorch)

PyTorch pipeline for reCAPTCHA tile classification.

## Folders

- `src/captcha_vision/` — package code (`data`, `models`, `training`, `common`)
- `data/dataset/` — train/test images
- `models/` — checkpoints
- `artifacts/logs/` — logs and evaluation outputs
- `scripts/` — runnable entry scripts
- `configs/` — default config files

## Run

```bash
source .venv/bin/activate
python scripts/train.py --data_dir data/dataset --output_dir models
python scripts/evaluate.py --checkpoint models/best_model.pt --data_dir data/dataset
```

