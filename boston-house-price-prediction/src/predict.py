"""
Predict a house price using a trained regression model.

Example:
    python src/predict.py --features "{\"RM\": 6.5, \"CRIM\": 0.2, \"TAX\": 300, \"PTRATIO\": 15, \"DIS\": 4.0}"
"""

import argparse
import json
import pickle
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Predict Boston house price.")
    parser.add_argument("--model", default="models/boston_house_price_model.pkl", help="Path to trained model.")
    parser.add_argument("--features", required=True, help="JSON object containing feature values.")
    return parser.parse_args()


def load_model(model_path):
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    with path.open("rb") as file:
        return pickle.load(file)


def parse_features(features_json):
    try:
        features = json.loads(features_json)
    except json.JSONDecodeError as exc:
        raise ValueError("Features must be a valid JSON object.") from exc

    if not isinstance(features, dict):
        raise ValueError("Features must be a JSON object with column names and values.")

    return pd.DataFrame([features])


def main():
    args = parse_args()
    model = load_model(args.model)
    input_data = parse_features(args.features)
    prediction = model.predict(input_data)[0]

    print("\nPrediction Result")
    print("=" * 60)
    print(f"Predicted house price: {prediction:.2f}")


if __name__ == "__main__":
    main()

