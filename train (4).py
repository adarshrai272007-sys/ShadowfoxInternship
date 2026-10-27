"""
Boston House Price Prediction

Trains and compares regression models, evaluates them (MSE, RMSE, MAE, R2,
cross-validated R2), optionally tunes the Random Forest with GridSearchCV,
saves the best pipeline plus metrics and plots.

Run with your dataset:
    python src/train.py --csv data/BostonHousing.csv

Run with the built-in synthetic demo data:
    python src/train.py

Extra options:
    python src/train.py --csv data/BostonHousing.csv --tune --cv 5
"""

import argparse
import json
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor


TARGET_CANDIDATES = ["MEDV", "medv", "Price", "price", "target", "Target"]


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Boston house price prediction model.")
    parser.add_argument("--csv", default=None, help="Path to housing CSV dataset.")
    parser.add_argument("--output", default="models/boston_house_price_model.pkl", help="Path to save trained model.")
    parser.add_argument("--output-dir", default="outputs", help="Folder for metrics.json and plots.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set size (0-1).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--cv", type=int, default=5, help="Cross-validation folds for R2 (0 disables).")
    parser.add_argument("--tune", action="store_true", help="Tune the Random Forest with GridSearchCV.")
    parser.add_argument("--save-demo-csv", default=None, help="Write the synthetic demo dataset to this CSV and exit.")
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
            raise FileNotFoundError(
                f"CSV file not found: {path}\n"
                "Place the Boston housing CSV there, or run without --csv to use synthetic demo data."
            )
        return pd.read_csv(path)

    print("No CSV supplied. Using synthetic demo data (results are NOT real house-price accuracy).")
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


def make_pipeline(features, model):
    # A fresh preprocessor for every model so pipelines never share fitted state.
    return Pipeline(steps=[("preprocessor", build_preprocessor(features)), ("model", model)])


def default_models(seed):
    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=seed, max_depth=6),
        "Random Forest": RandomForestRegressor(random_state=seed, n_estimators=150, max_depth=8),
    }


def tune_random_forest(features, x_train, y_train, seed):
    grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [6, 10, None],
        "model__min_samples_split": [2, 5],
        "model__min_samples_leaf": [1, 2],
    }
    search = GridSearchCV(
        make_pipeline(features, RandomForestRegressor(random_state=seed)),
        grid,
        cv=3,
        scoring="r2",
        n_jobs=-1,
    )
    search.fit(x_train, y_train)
    print(f"Best Random Forest parameters: {search.best_params_}")
    print(f"Best cross-validated R2 (train data): {search.best_score_:.3f}")
    return search.best_estimator_, search.best_params_


def evaluate(name, pipeline, x_test, y_test):
    predictions = pipeline.predict(x_test)
    mse = mean_squared_error(y_test, predictions)
    return {
        "name": name,
        "pipeline": pipeline,
        "predictions": predictions,
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_test, predictions)),
        "r2": float(r2_score(y_test, predictions)),
    }


def plot_model_comparison(results, path):
    names = [r["name"] for r in results]
    figure, (ax_r2, ax_rmse) = plt.subplots(1, 2, figsize=(10, 4))
    ax_r2.bar(names, [r["r2"] for r in results], color="#1A4A74")
    ax_r2.set_title("R2 score (higher is better)")
    ax_rmse.bar(names, [r["rmse"] for r in results], color="#7F9DB9")
    ax_rmse.set_title("RMSE (lower is better)")
    for axis in (ax_r2, ax_rmse):
        axis.tick_params(axis="x", rotation=20)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_actual_vs_predicted(y_test, predictions, title, path):
    figure, axis = plt.subplots(figsize=(5, 5))
    axis.scatter(y_test, predictions, alpha=0.7, color="#1A4A74")
    low, high = float(min(y_test.min(), predictions.min())), float(max(y_test.max(), predictions.max()))
    axis.plot([low, high], [low, high], "r--", label="Perfect prediction")
    axis.set_xlabel("Actual price")
    axis.set_ylabel("Predicted price")
    axis.set_title(title)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def feature_importances(pipeline):
    model = pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return None
    names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    clean = [str(n).split("__", 1)[-1] for n in names]
    return dict(sorted(zip(clean, map(float, model.feature_importances_)), key=lambda kv: -kv[1]))


def plot_feature_importance(importances, path):
    names, values = list(importances.keys())[::-1], list(importances.values())[::-1]
    figure, axis = plt.subplots(figsize=(6, 0.5 * len(names) + 1.5))
    axis.barh(names, values, color="#1A4A74")
    axis.set_title("Feature importance (best tree model)")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def save_model(model, output_path):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(model, file)
    print(f"\nSaved best model to: {path}")


def main():
    args = parse_args()

    if args.save_demo_csv:
        path = Path(args.save_demo_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        make_demo_dataset(seed=args.seed).to_csv(path, index=False)
        print(f"Wrote synthetic demo dataset to {path}")
        return

    dataframe = load_dataset(args.csv)
    target_column = find_target_column(dataframe)

    print("\nDataset Information")
    print("=" * 60)
    print(f"Rows and columns before cleaning: {dataframe.shape}")

    dataframe = dataframe.drop_duplicates().dropna(subset=[target_column])
    dataframe = remove_target_outliers_iqr(dataframe, target_column)

    print(f"Rows and columns after cleaning:  {dataframe.shape}")
    print(f"Target column: {target_column}")

    features = dataframe.drop(columns=[target_column])
    target = dataframe[target_column]

    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=args.test_size, random_state=args.seed
    )

    models = default_models(args.seed)
    results = []
    tuned_params = None

    for name, model in models.items():
        pipeline = make_pipeline(features, model).fit(x_train, y_train)
        result = evaluate(name, pipeline, x_test, y_test)
        if args.cv and args.cv > 1:
            scores = cross_val_score(make_pipeline(features, model), features, target, cv=args.cv, scoring="r2")
            result["cv_r2_mean"], result["cv_r2_std"] = float(scores.mean()), float(scores.std())
        results.append(result)

    if args.tune:
        print("\nTuning Random Forest with GridSearchCV ...")
        tuned_pipeline, tuned_params = tune_random_forest(features, x_train, y_train, args.seed)
        results.append(evaluate("Random Forest (tuned)", tuned_pipeline, x_test, y_test))

    print("\nModel Evaluation")
    print("=" * 90)
    for r in results:
        line = (
            f"{r['name']:<22} MSE: {r['mse']:.3f} | RMSE: {r['rmse']:.3f} | "
            f"MAE: {r['mae']:.3f} | R2: {r['r2']:.3f}"
        )
        if "cv_r2_mean" in r:
            line += f" | CV R2: {r['cv_r2_mean']:.3f} +/- {r['cv_r2_std']:.3f}"
        print(line)

    best = max(results, key=lambda item: item["r2"])
    print(f"\nBest model: {best['name']}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    importances = feature_importances(best["pipeline"])
    metrics = {
        "data_source": args.csv or "synthetic demo data",
        "rows_after_cleaning": int(len(dataframe)),
        "target_column": target_column,
        "best_model": best["name"],
        "tuned_random_forest_params": tuned_params,
        "models": {
            r["name"]: {k: v for k, v in r.items() if k in ("mse", "rmse", "mae", "r2", "cv_r2_mean", "cv_r2_std")}
            for r in results
        },
        "feature_importance": importances,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    plot_model_comparison(results, output_dir / "model_comparison.png")
    plot_actual_vs_predicted(y_test, best["predictions"], f"{best['name']}: actual vs predicted", output_dir / "actual_vs_predicted.png")
    if importances:
        plot_feature_importance(importances, output_dir / "feature_importance.png")
    print(f"Metrics and plots saved in: {output_dir}/")

    save_model(best["pipeline"], args.output)


if __name__ == "__main__":
    main()
