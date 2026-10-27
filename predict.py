"""
Predict the tag of a new image (or every image in a folder) with a trained model.

Run:
    python src/predict.py --image path/to/image.jpg
    python src/predict.py --image path/to/folder --top-k 3
"""

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


def parse_args():
    parser = argparse.ArgumentParser(description="Predict an image tag.")
    parser.add_argument("--image", required=True, help="Path to an image file or a folder of images.")
    parser.add_argument("--model", default="models/image_tagging_model.keras", help="Path to trained model.")
    parser.add_argument("--labels", default="models/class_names.json", help="Path to saved class labels.")
    parser.add_argument("--config", default="models/model_config.json", help="Path to saved model settings.")
    parser.add_argument("--image-size", type=int, default=None,
                        help="Image size used during training (default: read from the config file, else 180).")
    parser.add_argument("--top-k", type=int, default=3, help="Number of top tags to display.")
    parser.add_argument("--json", action="store_true", help="Print the result as JSON.")
    return parser.parse_args()


def load_labels(labels_path):
    path = Path(labels_path)
    if not path.exists():
        raise FileNotFoundError(f"Labels file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_image_size(cli_value, config_path):
    if cli_value:
        return cli_value
    path = Path(config_path)
    if path.exists():
        return int(json.loads(path.read_text(encoding="utf-8")).get("image_size", 180))
    return 180


def list_images(image_path):
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image path not found: {path}")
    if path.is_dir():
        files = sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
        if not files:
            raise ValueError(f"No images found in folder: {path}")
        return files
    return [path]


def load_image(image_path, image_size):
    image = tf.keras.utils.load_img(image_path, target_size=(image_size, image_size))
    return np.expand_dims(tf.keras.utils.img_to_array(image), axis=0)


def predict_image(model, image_array, class_names, top_k=3):
    probabilities = model.predict(image_array, verbose=0)[0]
    order = np.argsort(probabilities)[::-1]
    best = int(order[0])
    return {
        "predicted_tag": class_names[best],
        "confidence": float(probabilities[best]),
        "top_k": [
            {"tag": class_names[i], "probability": float(probabilities[i])}
            for i in order[: max(1, top_k)]
        ],
        "all_probabilities": {class_names[i]: float(p) for i, p in enumerate(probabilities)},
    }


def main():
    args = parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}. Train first with src/train.py.")

    model = tf.keras.models.load_model(model_path)
    class_names = load_labels(args.labels)
    image_size = resolve_image_size(args.image_size, args.config)

    results = []
    for image_file in list_images(args.image):
        result = predict_image(model, load_image(image_file, image_size), class_names, args.top_k)
        result["image"] = str(image_file)
        results.append(result)

    if args.json:
        print(json.dumps(results, indent=2))
        return

    for result in results:
        print("\nPrediction Result")
        print("=" * 60)
        print(f"Image:         {result['image']}")
        print(f"Predicted tag: {result['predicted_tag']}")
        print(f"Confidence:    {result['confidence']:.4f}")
        print(f"Top {len(result['top_k'])}:")
        for item in result["top_k"]:
            print(f"  {item['tag']:<25} {item['probability']:.4f}")


if __name__ == "__main__":
    main()
