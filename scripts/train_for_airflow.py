import argparse
import json
import os
import sys
import subprocess
import joblib
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from train import build_model, apply_scaling_if_needed, apply_smote


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_path", type=str, default="data/ci/train_sample.csv")
    p.add_argument("--test_path", type=str, default="data/ci/test_sample.csv")
    p.add_argument("--target", type=str, default="Class")
    p.add_argument("--model", type=str, default="rf")
    p.add_argument("--imbalance", type=str, default="none")
    p.add_argument("--scale_time_amount", action="store_true")
    p.add_argument("--n_estimators", type=int, default=100)
    p.add_argument("--max_depth", type=int, default=10)
    p.add_argument("--min_samples_leaf", type=int, default=1)
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--max_iter", type=int, default=1000)
    p.add_argument("--learning_rate", type=float, default=0.1)
    p.add_argument("--model_out", type=str, default="artifacts/model.pkl")
    p.add_argument("--metrics_out", type=str, default="artifacts/metrics.json")
    return p.parse_args()


def get_git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main():
    args = parse_args()

    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    os.makedirs(os.path.dirname(args.metrics_out), exist_ok=True)

    train_df = pd.read_csv(args.train_path)
    test_df = pd.read_csv(args.test_path)

    X_train = train_df.drop(columns=[args.target])
    y_train = train_df[args.target].astype(int)

    X_test = test_df.drop(columns=[args.target])
    y_test = test_df[args.target].astype(int)

    if args.scale_time_amount:
        X_train, X_test = apply_scaling_if_needed(X_train, X_test)

    if args.imbalance == "smote":
        X_train, y_train = apply_smote(X_train, y_train, random_state=42)

    model = build_model(args)
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    y_pred = (proba >= 0.5).astype(int)

    metrics = {
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "pr_auc": float(average_precision_score(y_test, proba)),
        "git_commit": get_git_commit(),
        "data_version": "ci_sample_v1",
        "seed": 42,
        "model_type": args.model,
    }

    joblib.dump(model, args.model_out)

    with open(args.metrics_out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()