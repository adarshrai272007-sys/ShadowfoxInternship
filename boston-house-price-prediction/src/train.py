"""
Boston House Price Prediction

This script trains and evaluates regression models for house price prediction.

Run with dataset:
    python src/train.py --csv data/BostonHousing.csv

Run with demo dataset:
    python src/train.py
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor


TARGET_CANDIDATES = ["MEDV", "medv", "Price", "price", "target", "Target"]


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Boston house price prediction model.")
    parser.add_argument("--csv", default=None, help="Path to housing CSV dataset.")
    parser.add_argument("--output", default="models/boston_house_price_model.pkl", help="Path to save trained model.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def make_demo_dataset(rows=300, seed=42):
    rng = np.random.default_rng(seed)
    rooms = rng.normal(6.2, 0.7, rows).clip(3.0, 9.0)
    crime_rate = rng.exponential(1.5, rows)
    tax = rng.normal(400, 80, rows).clip(180, 700)
    pupil_teacher_ratio = rng.normal(18, 2, rows).clip(12, 25)
    distance = rng.normal(4, 1.5, rows).clip(1, 12)

    price = (
        4.8 * rooms
        - 0.45 * crime_rate
        - 0.025 * tax
        - 0.8 * pupil_teacher_ratio
        + 0.7 * distance
        + rng.normal(0, 2.5, rows)
        + 25
    )

    return pd.DataFrame(
        {
            "RM": rooms,
            "CRIM": crime_rate,
            "TAX": tax,
            "PTRATIO": pupil_teacher_ratio,
            "DIS": distance,
            "MEDV": price.clip(5, 60),
        }
    )


def load_dataset(csv_path):
    if csv_path:
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")
        return pd.read_csv(path)

    print("No CSV supplied. Using synthetic demo data.")
    return make_demo_dataset()


def find_target_column(dataframe):
    for column in TARGET_CANDIDATES:
        if column in dataframe.columns:
            return column
    raise ValueError("Target column not found. Use one of: " + ", ".join(TARGET_CANDIDATES))


def remove_target_outliers_iqr(dataframe, target_column):
    q1 = dataframe[target_column].quantile(0.25)
    q3 = dataframe[target_column].quantile(0.75)
    iqr = q3 - q1
    lower_limit = q1 - 1.5 * iqr
    upper_limit = q3 + 1.5 * iqr
    return dataframe[
        (dataframe[target_column] >= lower_limit)
        & (dataframe[target_column] <= upper_limit)
    ]


def build_preprocessor(features):
    numeric_columns = features.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = features.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_columns),
            ("cat", categorical_pipeline, categorical_columns),
        ]
    )


def train_and_evaluate(name, model, preprocessor, x_train, x_test, y_train, y_test):
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)

    mse = mean_squared_error(y_test, predictions)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    return {
        "name": name,
        "pipeline": pipeline,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }


def save_model(model, output_path):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(model, file)
    print(f"\nSaved best model to: {path}")


def main():
    args = parse_args()
    dataframe = load_dataset(args.csv)
    target_column = find_target_column(dataframe)

    print("\nDataset Information")
    print("=" * 60)
    print(f"Rows and columns before cleaning: {dataframe.shape}")

    dataframe = dataframe.drop_duplicates()
    dataframe = remove_target_outliers_iqr(dataframe, target_column)

    print(f"Rows and columns after cleaning:  {dataframe.shape}")
    print(f"Target column: {target_column}")

    features = dataframe.drop(columns=[target_column])
    target = dataframe[target_column]

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=args.test_size,
        random_state=args.seed,
    )

    preprocessor = build_preprocessor(features)
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=args.seed, max_depth=6),
        "Random Forest": RandomForestRegressor(
            random_state=args.seed,
            n_estimators=150,
            max_depth=8,
        ),
    }

    results = []
    for name, model in models.items():
        results.append(
            train_and_evaluate(name, model, preprocessor, x_train, x_test, y_train, y_test)
        )

    print("\nModel Evaluation")
    print("=" * 80)
    for result in results:
        print(
            f"{result['name']:<18} "
            f"MSE: {result['mse']:.3f} | "
            f"RMSE: {result['rmse']:.3f} | "
            f"MAE: {result['mae']:.3f} | "
            f"R2: {result['r2']:.3f}"
        )

    best_result = max(results, key=lambda item: item["r2"])
    print(f"\nBest model: {best_result['name']}")
    save_model(best_result["pipeline"], args.output)


if __name__ == "__main__":
    main()

