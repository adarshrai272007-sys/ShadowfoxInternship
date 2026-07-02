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
  requirements.txt
  .gitignore
  src/
    train.py
    predict.py
  data/
    README.md
  models/
    README.md
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

