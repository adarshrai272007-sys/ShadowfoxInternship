"""
Create a small synthetic image dataset (circles, squares, triangles) so the
image tagging pipeline can be tested without downloading anything.

Run:
    python src/make_demo_dataset.py
    python src/make_demo_dataset.py --output-dir data/images --per-class 100

This is only a smoke-test dataset. For a real experiment use
src/download_dataset.py or your own labeled images.
"""

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw


CLASSES = ["circles", "squares", "triangles"]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a synthetic shapes dataset.")
    parser.add_argument("--output-dir", default="data/images", help="Where to write the images.")
    parser.add_argument("--per-class", type=int, default=100, help="Images per class.")
    parser.add_argument("--size", type=int, default=180, help="Image height and width.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def random_color(rng, low=0, high=255):
    return tuple(rng.randint(low, high) for _ in range(3))


def draw_shape(shape, size, rng):
    background = random_color(rng, 180, 255)
    foreground = random_color(rng, 0, 120)
    image = Image.new("RGB", (size, size), background)
    draw = ImageDraw.Draw(image)

    radius = rng.randint(size // 6, size // 3)
    cx = rng.randint(radius + 2, size - radius - 2)
    cy = rng.randint(radius + 2, size - radius - 2)

    if shape == "circles":
        draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=foreground)
    elif shape == "squares":
        draw.rectangle([cx - radius, cy - radius, cx + radius, cy + radius], fill=foreground)
    else:
        draw.polygon(
            [(cx, cy - radius), (cx - radius, cy + radius), (cx + radius, cy + radius)],
            fill=foreground,
        )
    return image


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    output_dir = Path(args.output_dir)

    for shape in CLASSES:
        class_dir = output_dir / shape
        class_dir.mkdir(parents=True, exist_ok=True)
        for index in range(args.per_class):
            draw_shape(shape, args.size, rng).save(class_dir / f"{shape}_{index:03d}.png")

    total = args.per_class * len(CLASSES)
    print(f"Created {total} images in {output_dir} ({', '.join(CLASSES)}).")


if __name__ == "__main__":
    main()
