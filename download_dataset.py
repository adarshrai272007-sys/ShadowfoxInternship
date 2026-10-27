"""
Download a real labeled image dataset (TensorFlow "flower_photos": 5 classes,
about 3,670 images) and place it in data/images.

Run:
    python src/download_dataset.py

Requires an internet connection. The archive is about 220 MB.
"""

import argparse
import shutil
from pathlib import Path

import tensorflow as tf

DATASET_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"


def parse_args():
    parser = argparse.ArgumentParser(description="Download the flower photos dataset.")
    parser.add_argument("--output-dir", default="data/images", help="Destination folder.")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)

    archive = tf.keras.utils.get_file("flower_photos", origin=DATASET_URL, untar=True)
    source = Path(archive)
    if not source.exists():
        source = Path(archive) / "flower_photos"

    output_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for class_dir in sorted(p for p in source.iterdir() if p.is_dir()):
        target = output_dir / class_dir.name
        target.mkdir(parents=True, exist_ok=True)
        for image_file in class_dir.glob("*.jpg"):
            shutil.copy2(image_file, target / image_file.name)
            copied += 1

    print(f"Copied {copied} images to {output_dir}.")


if __name__ == "__main__":
    main()
