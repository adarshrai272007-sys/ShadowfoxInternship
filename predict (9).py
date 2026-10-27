"""
Predict house prices using a trained regression model.

Single house (JSON text or JSON file):
    python src/predict.py --features "{\"RM\": 6.5, \"CRIM\": 0.2, \"TAX\": 300, \"PTRATIO\": 15, \"DIS\": 4.0}"
    python src/predict.py --features-file house.json

Many houses (CSV with the same feature columns):
    python src/predict.py --csv new_houses.csv --output predictions.csv
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Predict Boston house price.")
    parser.add_argument("--model", default="models/boston_house_price_model.pkl", help="Path to trained model.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--features", help="JSON object containing feature values.")
    source.add_argument("--features-file", help="Path to a JSON file with one object of feature values.")
    source.add_argument("--csv", help="CSV file with one house per row.")
    parser.add_argument("--output", default=None, help="Where to save batch predictions (CSV mode).")
    return parser.parse_args()


def load_model(model_path):
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}. Train first with src/train.py.")

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


def align_to_model(model, input_data):
    """Keep only columns the model was trained on; missing ones become NaN
    (the pipeline imputes them) and a warning is printed."""
    expected = list(getattr(model, "feature_names_in_", []))
    if not expected:
        return input_data

    unknown = [c for c in input_data.columns if c not in expected]
    missing = [c for c in expected if c not in input_data.columns]
    if unknown:
        print(f"Warning: ignoring unknown features: {unknown}", file=sys.stderr)
    if missing:
        print(f"Warning: missing features filled by the model's imputer: {missing}", file=sys.stderr)
    if len(missing) == len(expected):
        raise ValueError(f"None of the expected features were supplied. Expected: {expected}")
    return input_data.reindex(columns=expected)


def main():
    args = parse_args()
    model = load_model(args.model)

    if args.csv:
        data = pd.read_csv(args.csv)
        predictions = model.predict(align_to_model(model, data))
        data["predicted_price"] = predictions.round(2)
        if args.output:
            data.to_csv(args.output, index=False)
            print(f"Saved {len(data)} predictions to {args.output}")
        else:
            print(data.to_string(index=False))
        return

    text = Path(args.features_file).read_text(encoding="utf-8") if args.features_file else args.features
    input_data = align_to_model(model, parse_features(text))
    prediction = model.predict(input_data)[0]

    print("\nPrediction Result")
    print("=" * 60)
    print(f"Predicted house price: {prediction:.2f}")


if __name__ == "__main__":
    main()
