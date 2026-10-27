"""
Image Tagging with TensorFlow

Trains a CNN image classifier, evaluates it (accuracy, precision, recall,
confusion matrix), saves training curves and saves the model for prediction.

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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


def parse_args():
    parser = argparse.ArgumentParser(description="Train an image tagging CNN model.")
    parser.add_argument("--data-dir", default="data/images", help="Path to labeled image dataset.")
    parser.add_argument("--model-output", default="models/image_tagging_model.keras", help="Path to save model.")
    parser.add_argument("--labels-output", default="models/class_names.json", help="Path to save class names.")
    parser.add_argument("--config-output", default="models/model_config.json", help="Path to save model settings.")
    parser.add_argument("--output-dir", default="outputs", help="Folder for plots and metric files.")
    parser.add_argument("--image-size", type=int, default=180, help="Image height and width.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=10, help="Maximum number of training epochs.")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Adam learning rate.")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate before the output layer.")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (0 disables it).")
    parser.add_argument("--validation-split", type=float, default=0.2, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def validate_dataset(data_dir):
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {data_path}\n"
            "Create one with:  python src/make_demo_dataset.py   (synthetic)\n"
            "             or:  python src/download_dataset.py     (real photos)"
        )

    class_folders = [item for item in data_path.iterdir() if item.is_dir()]
    if len(class_folders) < 2:
        raise ValueError("Dataset must contain at least two class folders.")

    image_count = 0
    for folder in class_folders:
        image_count += sum(1 for file in folder.iterdir() if file.suffix.lower() in IMAGE_EXTENSIONS)

    if image_count == 0:
        raise ValueError("No image files found in the dataset folders.")


def load_datasets(args):
    common = dict(
        validation_split=args.validation_split,
        seed=args.seed,
        image_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
    )
    train_dataset = tf.keras.utils.image_dataset_from_directory(args.data_dir, subset="training", **common)
    validation_dataset = tf.keras.utils.image_dataset_from_directory(args.data_dir, subset="validation", **common)
    return train_dataset, validation_dataset, train_dataset.class_names


def prepare_datasets(train_dataset, validation_dataset):
    """Cache and prefetch only. Augmentation lives inside the model so that it is
    re-randomised every epoch (caching augmented images would freeze it)."""
    autotune = tf.data.AUTOTUNE
    train_dataset = train_dataset.cache().shuffle(1000, seed=42).prefetch(autotune)
    validation_dataset = validation_dataset.cache().prefetch(autotune)
    return train_dataset, validation_dataset


def build_cnn_model(image_size, number_of_classes, dropout=0.3, learning_rate=0.001):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(image_size, image_size, 3)),
            # Augmentation layers are only active during training.
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.Rescaling(1.0 / 255),
            tf.keras.layers.Conv2D(32, kernel_size=3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, kernel_size=3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(128, kernel_size=3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(number_of_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
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

        precision_scores.append(float(precision))
        recall_scores.append(float(recall))

    return precision_scores, recall_scores


def confusion_matrix(y_true, y_pred, number_of_classes):
    matrix = np.zeros((number_of_classes, number_of_classes), dtype=int)
    for true_label, predicted_label in zip(y_true, y_pred):
        matrix[true_label, predicted_label] += 1
    return matrix


def collect_predictions(model, dataset):
    true_labels, predicted_labels = [], []
    for images, labels in dataset:
        probabilities = model.predict(images, verbose=0)
        predicted_labels.extend(np.argmax(probabilities, axis=1))
        true_labels.extend(labels.numpy())
    return np.array(true_labels), np.array(predicted_labels)


def evaluate_model(model, validation_dataset, class_names):
    validation_loss, validation_accuracy = model.evaluate(validation_dataset, verbose=0)
    true_labels, predicted_labels = collect_predictions(model, validation_dataset)
    precision_scores, recall_scores = calculate_precision_recall(true_labels, predicted_labels, len(class_names))
    matrix = confusion_matrix(true_labels, predicted_labels, len(class_names))

    print("\nEvaluation Results")
    print("=" * 60)
    print(f"Validation loss:     {validation_loss:.4f}")
    print(f"Validation accuracy: {validation_accuracy:.4f}")
    print(f"Macro precision:     {np.mean(precision_scores):.4f}")
    print(f"Macro recall:        {np.mean(recall_scores):.4f}")
    print("\nClass-wise Metrics")
    print("-" * 60)
    for index, class_name in enumerate(class_names):
        print(f"{class_name:<25} Precision: {precision_scores[index]:.4f} | Recall: {recall_scores[index]:.4f}")

    return {
        "validation_loss": float(validation_loss),
        "validation_accuracy": float(validation_accuracy),
        "macro_precision": float(np.mean(precision_scores)),
        "macro_recall": float(np.mean(recall_scores)),
        "per_class": {
            name: {"precision": precision_scores[i], "recall": recall_scores[i]}
            for i, name in enumerate(class_names)
        },
        "confusion_matrix": matrix.tolist(),
    }


def plot_training_curves(history, output_path):
    figure, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(10, 4))
    ax_acc.plot(history["accuracy"], label="train")
    ax_acc.plot(history["val_accuracy"], label="validation")
    ax_acc.set_title("Accuracy")
    ax_acc.set_xlabel("Epoch")
    ax_acc.legend()
    ax_loss.plot(history["loss"], label="train")
    ax_loss.plot(history["val_loss"], label="validation")
    ax_loss.set_title("Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def plot_confusion_matrix(matrix, class_names, output_path):
    matrix = np.array(matrix)
    figure, axis = plt.subplots(figsize=(1.2 * len(class_names) + 3, 1.2 * len(class_names) + 2.5))
    axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(len(class_names)))
    axis.set_yticks(range(len(class_names)))
    axis.set_xticklabels(class_names, rotation=45, ha="right")
    axis.set_yticklabels(class_names)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title("Confusion matrix (validation set)")
    threshold = matrix.max() / 2 if matrix.max() else 0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            axis.text(col, row, int(matrix[row, col]), ha="center", va="center",
                      color="white" if matrix[row, col] > threshold else "black")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_outputs(model, class_names, metrics, history, args):
    model_path = Path(args.model_output)
    labels_path = Path(args.labels_output)
    config_path = Path(args.config_output)
    output_dir = Path(args.output_dir)

    for path in (model_path, labels_path, config_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.save(model_path)
    labels_path.write_text(json.dumps(class_names, indent=2), encoding="utf-8")
    config_path.write_text(json.dumps({"image_size": args.image_size}, indent=2), encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    plot_training_curves(history, output_dir / "training_curves.png")
    plot_confusion_matrix(metrics["confusion_matrix"], class_names, output_dir / "confusion_matrix.png")

    print("\nSaved Files")
    print("=" * 60)
    print(f"Model:            {model_path}")
    print(f"Labels:           {labels_path}")
    print(f"Model settings:   {config_path}")
    print(f"Metrics + plots:  {output_dir}/")


def main():
    args = parse_args()
    tf.keras.utils.set_random_seed(args.seed)

    validate_dataset(args.data_dir)
    train_dataset, validation_dataset, class_names = load_datasets(args)

    print("\nDetected Classes")
    print("=" * 60)
    for class_name in class_names:
        print(f"- {class_name}")

    train_dataset, validation_dataset = prepare_datasets(train_dataset, validation_dataset)
    model = build_cnn_model(args.image_size, len(class_names), args.dropout, args.learning_rate)

    print("\nModel Architecture")
    print("=" * 60)
    model.summary()

    callbacks = []
    if args.patience > 0:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.patience, restore_best_weights=True)
        )

    print("\nTraining Started")
    print("=" * 60)
    history = model.fit(train_dataset, validation_data=validation_dataset, epochs=args.epochs, callbacks=callbacks)

    metrics = evaluate_model(model, validation_dataset, class_names)
    save_outputs(model, class_names, metrics, history.history, args)


if __name__ == "__main__":
    main()
