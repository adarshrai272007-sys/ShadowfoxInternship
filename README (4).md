# Image Tagging with TensorFlow

This project trains a Convolutional Neural Network (CNN) to classify images into tags using TensorFlow and Keras.

## Objective

Develop a model for image tagging using TensorFlow.

The project includes:

- Data loading from labeled image folders
- Image preprocessing
- Data augmentation using rotation, flipping, and zoom
- CNN model architecture
- Model training with Keras
- Hyperparameter options through command-line arguments
- Evaluation using accuracy, precision, and recall
- Deployment-style prediction script for new images

## Project Structure

```text
image-tagging-tensorflow/
  README.md
  SUBMISSION.md
  requirements.txt
  src/
    train.py              # train, evaluate, save model + plots
    predict.py            # tag one image or a whole folder
    make_demo_dataset.py  # synthetic circles/squares/triangles (smoke test)
    download_dataset.py   # real flower photos dataset
  tests/
  data/README.md
  models/README.md
```

## Dataset Format

Place your images inside `data/images/` using one folder per class.

Example:

```text
data/images/
  cats/
    cat1.jpg
    cat2.jpg
  dogs/
    dog1.jpg
    dog2.jpg
```

The folder names become the image tags.

## Get a Dataset

Pick one:

```powershell
python src/make_demo_dataset.py     # 300 synthetic images, no download, for testing the pipeline
python src/download_dataset.py      # real photos (5 flower classes, ~220 MB, needs internet)
```

or copy your own labeled folders into `data/images/`.

## Installation

Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

## Train the Model

Run:

```powershell
python src/train.py --data-dir data/images --epochs 10
```

Optional hyperparameters:

```powershell
python src/train.py --data-dir data/images --epochs 20 --batch-size 16 --image-size 224
```

After training, the script saves:

```text
models/image_tagging_model.keras
models/class_names.json
```

## Predict a New Image

Run:

```powershell
python src/predict.py --image path/to/sample.jpg
```

You can also provide custom model paths:

```powershell
python src/predict.py --model models/image_tagging_model.keras --labels models/class_names.json --image path/to/sample.jpg
```

## Evaluation Metrics

The training script prints:

- Validation accuracy
- Class-wise precision
- Class-wise recall
- Macro precision
- Macro recall

## Notes for Submission

This project satisfies the task requirements:

- A labeled image dataset is used.
- A CNN model is designed for image classification.
- The model is trained using TensorFlow Keras.
- Data augmentation is applied.
- Hyperparameters can be tuned from the command line.
- Model performance is evaluated using accuracy, precision, and recall.
- A prediction script is included for deployment-style usage.


## Outputs

After training the script writes:

```text
models/image_tagging_model.keras   trained model
models/class_names.json            tag names
models/model_config.json           image size used for training
outputs/metrics.json               accuracy, macro precision/recall, per-class metrics, confusion matrix
outputs/training_curves.png        accuracy and loss per epoch
outputs/confusion_matrix.png       validation confusion matrix
```

More options: `--learning-rate`, `--dropout`, `--patience` (early stopping, 0 disables), `--config-output`, `--output-dir`.

## Predict

```powershell
python src/predict.py --image path/to/image.jpg
python src/predict.py --image path/to/folder --top-k 3
python src/predict.py --image path/to/image.jpg --json
```

## Tests

```powershell
python -m pytest -q
```
