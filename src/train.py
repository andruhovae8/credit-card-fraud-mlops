import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve
)
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import mlflow
import mlflow.sklearn


def parse_args():
    p = argparse.ArgumentParser()

    # prepared data (from DVC prepare stage)
    p.add_argument("--train_path", type=str, default="data/prepared/train.csv")
    p.add_argument("--test_path", type=str, default="data/prepared/test.csv")
    p.add_argument("--target", type=str, default="Class")

    # preprocessing
    p.add_argument("--scale_time_amount", action="store_true",
                   help="Scale only Time and Amount with RobustScaler (recommended for linear models)")

    # imbalance
    p.add_argument("--imbalance", type=str, default="none",
                   choices=["none", "smote", "class_weight"],
                   help="Imbalance handling method")

    # model
    p.add_argument("--model", type=str, default="rf",
                   choices=["rf", "lr", "xgb", "lgbm"])

    # RF params
    p.add_argument("--n_estimators", type=int, default=300)
    p.add_argument("--max_depth", type=int, default=None)
    p.add_argument("--min_samples_leaf", type=int, default=1)

    # LR params
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--max_iter", type=int, default=1000)

    # XGB / LGBM basic params
    p.add_argument("--learning_rate", type=float, default=0.1)

    # threshold
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Decision threshold for converting proba -> class")
    p.add_argument("--tune_threshold", action="store_true",
                   help="Tune threshold on TRAIN (used) set by maximizing F2-score. Logs best_threshold.")

    # SHAP (optional)
    p.add_argument("--shap", action="store_true")
    p.add_argument("--shap_sample", type=int, default=1000)

    # mlflow
    p.add_argument("--experiment_name", type=str, default="MLOps_Lab_1_Fraud")
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--author", type=str, default="student")
    p.add_argument("--dataset_version", type=str, default="creditcard_v1")

    return p.parse_args()


def save_confusion_matrix(y_true, y_pred, out_path):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(values_format="d")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    return out_path


def save_feature_importance(model, feature_names, out_path, top_k=15):
    # Tree models feature_importances_
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        idx = np.argsort(imp)[::-1][:top_k]
        plt.figure(figsize=(8, 4))
        plt.bar(range(len(idx)), imp[idx])
        plt.xticks(range(len(idx)), [feature_names[i] for i in idx], rotation=60, ha="right")
        plt.title("Top Feature Importances")
        plt.tight_layout()
        plt.savefig(out_path, dpi=160)
        plt.close()
        return out_path

    # Logistic Regression coefficients
    if hasattr(model, "coef_"):
        coefs = model.coef_.ravel()
        imp = np.abs(coefs)
        idx = np.argsort(imp)[::-1][:top_k]
        plt.figure(figsize=(8, 4))
        plt.bar(range(len(idx)), imp[idx])
        plt.xticks(range(len(idx)), [feature_names[i] for i in idx], rotation=60, ha="right")
        plt.title("Top features by |coef|")
        plt.tight_layout()
        plt.savefig(out_path, dpi=160)
        plt.close()
        return out_path

    return None


def build_model(args):
    if args.model == "rf":
        cw = None
        if args.imbalance == "class_weight":
            cw = "balanced_subsample"

        return RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            class_weight=cw,
            random_state=42,
            n_jobs=-1,
        )

    if args.model == "lr":
        cw = None
        if args.imbalance == "class_weight":
            cw = "balanced"

        return LogisticRegression(
            C=args.C,
            class_weight=cw,
            max_iter=args.max_iter,
            solver="lbfgs",
        )

    if args.model == "xgb":
        try:
            from xgboost import XGBClassifier
        except ImportError as e:
            raise ImportError("xgboost is not installed. Run: pip install xgboost") from e

        return XGBClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth if args.max_depth is not None else 6,
            learning_rate=args.learning_rate,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            eval_metric="logloss",
            n_jobs=-1,
        )

    if args.model == "lgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as e:
            raise ImportError("lightgbm is not installed. Run: pip install lightgbm") from e

        return LGBMClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth if args.max_depth is not None else -1,
            learning_rate=args.learning_rate,
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        )

    raise ValueError("Unknown model")


def apply_scaling_if_needed(X_train, X_test, scale_cols=("Time", "Amount")):
    scaler = RobustScaler()
    X_train = X_train.copy()
    X_test = X_test.copy()

    cols = [c for c in scale_cols if c in X_train.columns]
    if cols:
        X_train.loc[:, cols] = scaler.fit_transform(X_train[cols])
        X_test.loc[:, cols] = scaler.transform(X_test[cols])

    return X_train, X_test


def apply_smote(X_train, y_train, random_state=42):
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError as e:
        raise ImportError("imbalanced-learn is not installed. Run: pip install imbalanced-learn") from e

    smote = SMOTE(random_state=random_state, k_neighbors=5)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    return X_res, y_res


def compute_metrics(y_true, y_pred, y_proba):
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "pr_auc": average_precision_score(y_true, y_proba),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def tune_threshold_f2(y_true, y_proba):
    precision_vals, recall_vals, thresholds = precision_recall_curve(y_true, y_proba)
    precision_vals = precision_vals[:-1]
    recall_vals = recall_vals[:-1]

    f2 = (5 * precision_vals * recall_vals) / (4 * precision_vals + recall_vals + 1e-12)
    best_idx = int(np.argmax(f2))
    return float(thresholds[best_idx]), float(f2[best_idx])


def log_shap_if_requested(args, model, X_test):
    if not args.shap:
        return

    if args.model not in ["rf", "xgb", "lgbm"]:
        return

    try:
        import shap
    except ImportError as e:
        raise ImportError("shap is not installed. Run: pip install shap") from e

    n = min(args.shap_sample, len(X_test))
    Xs = X_test.sample(n=n, random_state=42)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(Xs)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    out_path = "shap_summary.png"
    plt.figure()
    shap.summary_plot(shap_values, Xs, show=False, plot_type="bar")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    mlflow.log_artifact(out_path)


def main():
    args = parse_args()

    train_df = pd.read_csv(args.train_path)
    test_df = pd.read_csv(args.test_path)

    if args.target not in train_df.columns:
        raise ValueError(f"Target '{args.target}' not found in train data")
    if args.target not in test_df.columns:
        raise ValueError(f"Target '{args.target}' not found in test data")

    X_train = train_df.drop(columns=[args.target])
    y_train = train_df[args.target].astype(int)

    X_test = test_df.drop(columns=[args.target])
    y_test = test_df[args.target].astype(int)

    feature_names = X_train.columns.tolist()

    # optional scaling (mostly for LR)
    if args.scale_time_amount:
        X_train, X_test = apply_scaling_if_needed(X_train, X_test)

    # imbalance handling
    X_train_used, y_train_used = X_train, y_train
    if args.imbalance == "smote":
        X_train_used, y_train_used = apply_smote(X_train, y_train, random_state=42)

    model = build_model(args)

    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run(run_name=args.run_name):
        # tags
        mlflow.set_tag("author", args.author)
        mlflow.set_tag("dataset_version", args.dataset_version)
        mlflow.set_tag("model_type", args.model)
        mlflow.set_tag("imbalance", args.imbalance)
        mlflow.set_tag("task", "fraud_detection")

        # params (general)
        mlflow.log_param("train_path", args.train_path)
        mlflow.log_param("test_path", args.test_path)
        mlflow.log_param("scale_time_amount", bool(args.scale_time_amount))
        mlflow.log_param("threshold", args.threshold)
        mlflow.log_param("tune_threshold", bool(args.tune_threshold))

        # params (model-specific)
        if args.model == "rf":
            mlflow.log_param("n_estimators", args.n_estimators)
            mlflow.log_param("max_depth", args.max_depth)
            mlflow.log_param("min_samples_leaf", args.min_samples_leaf)

        if args.model == "lr":
            mlflow.log_param("C", args.C)
            mlflow.log_param("max_iter", args.max_iter)

        if args.model in ["xgb", "lgbm"]:
            mlflow.log_param("n_estimators", args.n_estimators)
            mlflow.log_param("max_depth", args.max_depth)
            mlflow.log_param("learning_rate", args.learning_rate)

        # train on train_used
        model.fit(X_train_used, y_train_used)

        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")
        mlflow.log_artifact("models/model.pkl")

        # proba: TRAIN metrics computed on train_used, TEST on test
        if hasattr(model, "predict_proba"):
            proba_train = model.predict_proba(X_train_used)[:, 1]
            proba_test = model.predict_proba(X_test)[:, 1]
        else:
            proba_train = model.decision_function(X_train_used)
            proba_test = model.decision_function(X_test)

        used_threshold = args.threshold
        if args.tune_threshold:
            best_thr, best_f2 = tune_threshold_f2(y_train_used, proba_train)
            used_threshold = best_thr
            mlflow.log_param("best_threshold", best_thr)
            mlflow.log_metric("best_f2_train", best_f2)

        y_pred_train = (proba_train >= used_threshold).astype(int)
        y_pred_test = (proba_test >= used_threshold).astype(int)

        m_train = compute_metrics(y_train_used, y_pred_train, proba_train)
        m_test = compute_metrics(y_test, y_pred_test, proba_test)

        # metrics
        for k, v in m_train.items():
            mlflow.log_metric(f"{k}_train", float(v))
        for k, v in m_test.items():
            mlflow.log_metric(f"{k}_test", float(v))

        # artifacts
        cm_path = save_confusion_matrix(y_test, y_pred_test, "confusion_matrix.png")
        mlflow.log_artifact(cm_path)

        fi_path = save_feature_importance(model, feature_names, "feature_importance.png", top_k=15)
        if fi_path:
            mlflow.log_artifact(fi_path)

        log_shap_if_requested(args, model, X_test)

        # model
        mlflow.sklearn.log_model(model, name="model")

        print("Done.")
        print(
            f"Test: precision={m_test['precision']:.4f}, recall={m_test['recall']:.4f}, "
            f"f1={m_test['f1']:.4f}, pr_auc={m_test['pr_auc']:.4f}, roc_auc={m_test['roc_auc']:.4f}"
        )
        print(f"Used threshold: {used_threshold:.4f}")


if __name__ == "__main__":
    main()