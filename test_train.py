import json
import pickle
import subprocess
import sys
from pathlib import Path

import pandas as pd

from train import (find_target_column, make_demo_dataset, make_pipeline,
                   remove_target_outliers_iqr)

ROOT = Path(__file__).resolve().parents[1]


def test_demo_dataset_shape_and_target():
    data = make_demo_dataset(rows=50)
    assert data.shape == (50, 6)
    assert find_target_column(data) == "MEDV"


def test_missing_target_column_raises():
    try:
        find_target_column(pd.DataFrame({"a": [1]}))
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_iqr_removes_extreme_target_values():
    data = pd.DataFrame({"x": range(11), "MEDV": [10] * 10 + [1000]})
    cleaned = remove_target_outliers_iqr(data, "MEDV")
    assert len(cleaned) == 10


def test_pipeline_handles_missing_values():
    data = make_demo_dataset(rows=80)
    features, target = data.drop(columns="MEDV"), data["MEDV"]
    features.loc[0, "RM"] = None
    from sklearn.linear_model import LinearRegression
    model = make_pipeline(features, LinearRegression()).fit(features, target)
    assert len(model.predict(features)) == 80


def test_train_and_predict_cli(tmp_path):
    model = tmp_path / "m.pkl"
    out = tmp_path / "out"
    subprocess.run([sys.executable, str(ROOT / "src/train.py"), "--output", str(model),
                    "--output-dir", str(out), "--cv", "0"], check=True, capture_output=True)
    assert model.exists()
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["best_model"] in metrics["models"]

    result = subprocess.run(
        [sys.executable, str(ROOT / "src/predict.py"), "--model", str(model),
         "--features", '{"RM": 6.5, "CRIM": 0.2, "TAX": 300, "PTRATIO": 15, "DIS": 4.0}'],
        check=True, capture_output=True, text=True)
    assert "Predicted house price" in result.stdout
    assert pickle.loads(model.read_bytes()) is not None
