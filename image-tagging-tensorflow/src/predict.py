"""
Predict the tag of a new image using a trained TensorFlow model.

Run:
    python src/predict.py --image path/to/image.jpg
"""

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf


def parse_args():
    parser = argparse.ArgumentParser(description="Predict an image tag.")
    parser.add_argument("--image", required=True, help="Path to image file.")
    parser.add_argument("--model", default="models/image_tagging_model.keras", help="Path to trained model.")
    parser.add_argument("--labels", default="models/class_names.json", help="Path to saved class labels.")
    parser.add_argument("--image-size", type=int, default=180, help="Image height and width used during training.")
    return parser.parse_args()


def load_labels(labels_path):
    path = Path(labels_path)
    if not path.exists():
        raise FileNotFoundError(f"Labels file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_image(image_path, image_size):
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    image = tf.keras.utils.load_img(path, target_size=(image_size, image_size))
    image_array = tf.keras.utils.img_to_array(image)
    return np.expand_dims(image_array, axis=0)


def predict_image(model, image_array, class_names):
    probabilities = model.predict(image_array, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))

    return {
        "predicted_tag": class_names[predicted_index],
        "confidence": float(probabilities[predicted_index]),
        "all_probabilities": {
            class_names[index]: float(probability)
            for index, probability in enumerate(probabilities)
        },
    }


def main():
    args = parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = tf.keras.models.load_model(model_path)
    class_names = load_labels(args.labels)
    image_array = load_image(args.image, args.image_size)
    result = predict_image(model, image_array, class_names)

    print("\nPrediction Result")
    print("=" * 60)
    print(f"Image:         {args.image}")
    print(f"Predicted tag: {result['predicted_tag']}")
    print(f"Confidence:    {result['confidence']:.4f}")

    print("\nClass Probabilities")
    print("-" * 60)
    for class_name, probability in result["all_probabilities"].items():
        print(f"{class_name:<25} {probability:.4f}")


if __name__ == "__main__":
    main()

