import argparse
import os
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    average_precision_score,
)

from train import build_model, apply_scaling_if_needed, apply_smote


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--train_path", type=str, required=True)
    p.add_argument("--test_path", type=str, required=True)
    p.add_argument("--target", type=str, default="Class")
    p.add_argument("--model", type=str, default="rf")

    p.add_argument("--max_rows", type=int, default=5000)

    p.add_argument("--imbalance", type=str, default="none")
    p.add_argument("--scale_time_amount", action="store_true")

    # Параметри сумісності з build_model() із train.py
    p.add_argument("--n_estimators", type=int, default=100)
    p.add_argument("--max_depth", type=int, default=10)
    p.add_argument("--min_samples_leaf", type=int, default=1)

    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--max_iter", type=int, default=1000)

    p.add_argument("--learning_rate", type=float, default=0.1)

    p.add_argument("--model_out", type=str, default="artifacts/model.pkl")
    p.add_argument("--metrics_path", type=str, default="artifacts/metrics.json")
    p.add_argument("--cm_path", type=str, default="artifacts/confusion_matrix.png")

    return p.parse_args()


def compute_metrics(y_true, y_pred, y_proba):
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def save_confusion_matrix(y_true, y_pred, path):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5, 4))
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main():
    args = parse_args()

    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    os.makedirs(os.path.dirname(args.metrics_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.cm_path), exist_ok=True)

    train_df = pd.read_csv(args.train_path).head(args.max_rows)
    test_df = pd.read_csv(args.test_path).head(args.max_rows)

    if args.target not in train_df.columns or args.target not in test_df.columns:
        raise ValueError(f"Target column '{args.target}' not found")

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

    metrics = compute_metrics(y_test, y_pred, proba)

    joblib.dump(model, args.model_out)

    with open(args.metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    save_confusion_matrix(y_test, y_pred, args.cm_path)

    print("Metrics:", metrics)
    print(f"Saved model to: {args.model_out}")
    print(f"Saved metrics to: {args.metrics_path}")
    print(f"Saved confusion matrix to: {args.cm_path}")


if __name__ == "__main__":
    main()