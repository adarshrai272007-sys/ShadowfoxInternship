"""
Image Tagging with TensorFlow

This script trains a CNN image classification model using TensorFlow Keras.

Dataset format:
    data/images/
      class_1/
        image1.jpg
      class_2/
        image2.jpg

Run:
    python src/train.py --data-dir data/images --epochs 10
"""

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf


def parse_args():
    parser = argparse.ArgumentParser(description="Train an image tagging CNN model.")
    parser.add_argument("--data-dir", default="data/images", help="Path to labeled image dataset.")
    parser.add_argument("--model-output", default="models/image_tagging_model.keras", help="Path to save model.")
    parser.add_argument("--labels-output", default="models/class_names.json", help="Path to save class names.")
    parser.add_argument("--image-size", type=int, default=180, help="Image height and width.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--validation-split", type=float, default=0.2, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def validate_dataset(data_dir):
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_path}")

    class_folders = [item for item in data_path.iterdir() if item.is_dir()]
    if len(class_folders) < 2:
        raise ValueError("Dataset must contain at least two class folders.")

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
    image_count = 0
    for folder in class_folders:
        image_count += sum(1 for file in folder.iterdir() if file.suffix.lower() in image_extensions)

    if image_count == 0:
        raise ValueError("No image files found in the dataset folders.")


def load_datasets(args):
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        args.data_dir,
        validation_split=args.validation_split,
        subset="training",
        seed=args.seed,
        image_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
    )

    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        args.data_dir,
        validation_split=args.validation_split,
        subset="validation",
        seed=args.seed,
        image_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
    )

    class_names = train_dataset.class_names
    return train_dataset, validation_dataset, class_names


def prepare_datasets(train_dataset, validation_dataset):
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
        ],
        name="data_augmentation",
    )

    autotune = tf.data.AUTOTUNE
    train_dataset = train_dataset.map(
        lambda images, labels: (data_augmentation(images, training=True), labels)
    )
    train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=autotune)
    validation_dataset = validation_dataset.cache().prefetch(buffer_size=autotune)

    return train_dataset, validation_dataset


def build_cnn_model(image_size, number_of_classes):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(image_size, image_size, 3)),
            tf.keras.layers.Rescaling(1.0 / 255),
            tf.keras.layers.Conv2D(32, kernel_size=3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, kernel_size=3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(128, kernel_size=3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(number_of_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def calculate_precision_recall(y_true, y_pred, number_of_classes):
    precision_scores = []
    recall_scores = []

    for class_index in range(number_of_classes):
        true_positive = np.sum((y_true == class_index) & (y_pred == class_index))
        false_positive = np.sum((y_true != class_index) & (y_pred == class_index))
        false_negative = np.sum((y_true == class_index) & (y_pred != class_index))

        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0

        precision_scores.append(precision)
        recall_scores.append(recall)

    return precision_scores, recall_scores


def evaluate_model(model, validation_dataset, class_names):
    validation_loss, validation_accuracy = model.evaluate(validation_dataset, verbose=0)

    true_labels = []
    predicted_labels = []

    for images, labels in validation_dataset:
        probabilities = model.predict(images, verbose=0)
        predictions = np.argmax(probabilities, axis=1)
        true_labels.extend(labels.numpy())
        predicted_labels.extend(predictions)

    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)

    precision_scores, recall_scores = calculate_precision_recall(
        true_labels,
        predicted_labels,
        len(class_names),
    )

    print("\nEvaluation Results")
    print("=" * 60)
    print(f"Validation loss:     {validation_loss:.4f}")
    print(f"Validation accuracy: {validation_accuracy:.4f}")
    print(f"Macro precision:     {np.mean(precision_scores):.4f}")
    print(f"Macro recall:        {np.mean(recall_scores):.4f}")
    print("\nClass-wise Metrics")
    print("-" * 60)

    for index, class_name in enumerate(class_names):
        print(
            f"{class_name:<25} "
            f"Precision: {precision_scores[index]:.4f} | "
            f"Recall: {recall_scores[index]:.4f}"
        )


def save_outputs(model, class_names, model_output, labels_output):
    model_path = Path(model_output)
    labels_path = Path(labels_output)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.parent.mkdir(parents=True, exist_ok=True)

    model.save(model_path)
    with labels_path.open("w", encoding="utf-8") as file:
        json.dump(class_names, file, indent=2)

    print("\nSaved Files")
    print("=" * 60)
    print(f"Model saved to:  {model_path}")
    print(f"Labels saved to: {labels_path}")


def main():
    args = parse_args()

    validate_dataset(args.data_dir)
    train_dataset, validation_dataset, class_names = load_datasets(args)

    print("\nDetected Classes")
    print("=" * 60)
    for class_name in class_names:
        print(f"- {class_name}")

    train_dataset, validation_dataset = prepare_datasets(train_dataset, validation_dataset)
    model = build_cnn_model(args.image_size, len(class_names))

    print("\nModel Architecture")
    print("=" * 60)
    model.summary()

    print("\nTraining Started")
    print("=" * 60)
    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=args.epochs,
    )

    evaluate_model(model, validation_dataset, class_names)
    save_outputs(model, class_names, args.model_output, args.labels_output)


if __name__ == "__main__":
    main()

