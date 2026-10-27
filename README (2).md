# ShadowfoxInternship

# AIML Beginner Level Tasks

Complete beginner-level AI/ML task solutions. Every project has runnable source code, tests, a project README and a submission write-up.

## Projects Included

1. [Image Tagging with TensorFlow](image-tagging-tensorflow/)
2. [Boston House Price Prediction](boston-house-price-prediction/)
3. [Autocorrect Keyboard System](autocorrect-keyboard-system/)

## Repository Structure

```text
ShadowfoxInternship/
  README.md
  requirements.txt
  .gitignore
  .github/workflows/tests.yml
  image-tagging-tensorflow/
    src/  train.py  predict.py  make_demo_dataset.py  download_dataset.py
    tests/  README.md  SUBMISSION.md  requirements.txt  data/  models/
  boston-house-price-prediction/
    src/  train.py  predict.py
    tests/  README.md  SUBMISSION.md  requirements.txt  data/  models/
  autocorrect-keyboard-system/
    src/  autocorrect.py
    tests/  README.md  SUBMISSION.md  requirements.txt
```

## Quick Start

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 1. Image Tagging with TensorFlow

```powershell
cd image-tagging-tensorflow
python src/make_demo_dataset.py            # tiny synthetic shapes dataset (smoke test)
python src/train.py --data-dir data/images --epochs 15
python src/predict.py --image data/images/squares/squares_000.png
```

For real photos run `python src/download_dataset.py` (TensorFlow flower photos, ~220 MB) or put your own folders in `data/images/<class_name>/`.
Training saves the model, class names, `outputs/metrics.json`, `outputs/training_curves.png` and `outputs/confusion_matrix.png`.

### 2. Boston House Price Prediction

```powershell
cd boston-house-price-prediction
python src/train.py --csv data/BostonHousing.csv --tune --cv 5
python src/predict.py --features "{\"RM\": 6.5, \"CRIM\": 0.2, \"TAX\": 300, \"PTRATIO\": 15, \"DIS\": 4.0}"
```

Without a CSV, `python src/train.py` uses a synthetic demo dataset (a pipeline check only, not real accuracy).

### 3. Autocorrect Keyboard System

```powershell
cd autocorrect-keyboard-system
python src/autocorrect.py --text "I havv a beutiful hous"
python src/autocorrect.py --interactive
```

## Run the Tests

```powershell
cd autocorrect-keyboard-system; python -m pytest -q
cd ..\boston-house-price-prediction; python -m pytest -q
cd ..\image-tagging-tensorflow; python -m pytest -q
```

## Notes

Datasets, trained models and result files are generated locally and ignored by Git (see `.gitignore`).
