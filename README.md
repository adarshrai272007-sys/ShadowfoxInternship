# ShadowfoxInternship
# AIML Beginner Level Tasks

This repository contains complete beginner-level AIML task solutions.

## Projects Included

1. Image Tagging with TensorFlow
2. Boston House Price Prediction
3. Autocorrect Keyboard System

Each project includes source code, setup instructions, a submission writeup, and a project-specific README.

## Repository Structure

```text
aiml-beginner-tasks-repository/
  README.md
  requirements.txt
  .gitignore
  projects/
    image-tagging-tensorflow/
    boston-house-price-prediction/
    autocorrect-keyboard-system/
```

## 1. Image Tagging with TensorFlow

Folder:

```text
projects/image-tagging-tensorflow/
```

Objective:

Develop a CNN model for image tagging using TensorFlow.

Main files:

```text
projects/image-tagging-tensorflow/src/train.py
projects/image-tagging-tensorflow/src/predict.py
```

Run:

```powershell
cd projects/image-tagging-tensorflow
pip install -r requirements.txt
python src/train.py --data-dir data/images --epochs 10
```

## 2. Boston House Price Prediction

Folder:

```text
projects/boston-house-price-prediction/
```

Objective:

Train regression models to predict house prices using housing attributes.

Main files:

```text
projects/boston-house-price-prediction/src/train.py
projects/boston-house-price-prediction/src/predict.py
```

Run:

```powershell
cd projects/boston-house-price-prediction
pip install -r requirements.txt
python src/train.py --csv data/BostonHousing.csv
```

If no dataset is available, run:

```powershell
python src/train.py
```

## 3. Autocorrect Keyboard System

Folder:

```text
projects/autocorrect-keyboard-system/
```

Objective:

Detect misspelled words and suggest corrections.

Main file:

```text
projects/autocorrect-keyboard-system/src/autocorrect.py
```

Run:

```powershell
cd projects/autocorrect-keyboard-system
pip install -r requirements.txt
python src/autocorrect.py --text "I havv a beutiful hous"
```

## Installing All Requirements

You can install all common dependencies from the repository root:

```powershell
pip install -r requirements.txt
```

TensorFlow can take time to install. If you only want to run the non-image tasks first, install the project-specific requirements from those folders.

## GitHub Upload Instructions

1. Create a new GitHub repository.
2. Choose the `Python` `.gitignore` template if GitHub asks.
3. Upload the contents of this folder.
4. Keep datasets and trained models out of GitHub unless your instructor specifically asks for them.

## Notes

Datasets and trained models are ignored by Git because they can be large or generated locally.

