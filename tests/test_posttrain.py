import json
import os


def test_model_exists():
    assert os.path.exists("artifacts/model.pkl")


def test_metrics_exists():
    assert os.path.exists("artifacts/metrics.json")


def test_metrics_quality():
    with open("artifacts/metrics.json") as f:
        metrics = json.load(f)

    assert metrics["roc_auc"] > 0.8
    assert metrics["f1"] >= 0.5