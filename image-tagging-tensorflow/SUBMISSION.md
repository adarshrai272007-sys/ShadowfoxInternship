# Task 1: Image Tagging with TensorFlow

## Objective

The objective of this project is to develop an image tagging model using TensorFlow. The model learns from a labeled image dataset and predicts the correct tag for unseen images.

## Data Preparation

The dataset is arranged in folders where each folder represents one class label.

Example:

```text
data/images/
  cats/
    cat1.jpg
  dogs/
    dog1.jpg
```

TensorFlow's `image_dataset_from_directory` function is used to load the dataset. Images are resized to a fixed size before training. The dataset is split into training and validation sets.

## Model Architecture

The model is a Convolutional Neural Network built using TensorFlow Keras.

Layers used:

- Rescaling layer
- Conv2D layer with 32 filters
- MaxPooling2D layer
- Conv2D layer with 64 filters
- MaxPooling2D layer
- Conv2D layer with 128 filters
- MaxPooling2D layer
- Flatten layer
- Dense layer with 128 neurons
- Dropout layer
- Dense output layer with softmax activation

## Model Training

The model is trained using:

- Optimizer: Adam
- Loss function: Sparse categorical crossentropy
- Metric: Accuracy

Training command:

```powershell
python src/train.py --data-dir data/images --epochs 10
```

## Data Augmentation

The project uses the following augmentation methods:

- Random horizontal flipping
- Random rotation
- Random zoom

Data augmentation helps the model generalize better and reduces overfitting.

## Hyperparameter Tuning

The project supports tuning through command-line arguments:

- `--epochs`
- `--batch-size`
- `--image-size`
- `--validation-split`

Example:

```powershell
python src/train.py --data-dir data/images --epochs 20 --batch-size 16 --image-size 224
```

## Model Evaluation

The model is evaluated using:

- Validation accuracy
- Precision
- Recall

Precision measures how many predicted images for a class are correct. Recall measures how many actual images from a class are correctly identified.

## Deployment

The trained model is saved in the `models/` folder. A separate prediction script is included to classify new images.

Prediction command:

```powershell
python src/predict.py --image path/to/sample.jpg
```

## Conclusion

This project completes the full image tagging workflow: dataset preparation, CNN design, training, augmentation, hyperparameter tuning, evaluation, and deployment-style prediction. The model can be improved further by adding more training images or using transfer learning with pretrained models.

