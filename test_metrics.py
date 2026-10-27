import numpy as np
import pytest

pytest.importorskip("tensorflow")
from train import calculate_precision_recall, confusion_matrix  # noqa: E402


def test_precision_recall_known_values():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0])
    precision, recall = calculate_precision_recall(y_true, y_pred, 3)
    assert precision == pytest.approx([0.5, 2 / 3, 1.0])
    assert recall == pytest.approx([0.5, 1.0, 0.5])


def test_confusion_matrix_counts():
    matrix = confusion_matrix(np.array([0, 1, 1]), np.array([0, 0, 1]), 2)
    assert matrix.tolist() == [[1, 0], [1, 1]]
