import os
import joblib
import optuna
import hydra
import mlflow
import mlflow.sklearn
import pandas as pd

from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split

from train import (
    build_model,
    apply_scaling_if_needed,
    apply_smote,
    compute_metrics,
    tune_threshold_f2,
)

SEED = 42


class ArgsNamespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def make_args_from_cfg(cfg: DictConfig, trial: optuna.Trial | None = None):
    model_name = cfg.model.name

    params = {
        "train_path": cfg.data.train_path,
        "test_path": cfg.data.test_path,
        "target": cfg.data.target,
        "scale_time_amount": cfg.preprocessing.scale_time_amount,
        "imbalance": cfg.preprocessing.imbalance,
        "model": model_name,
        "threshold": cfg.threshold.value,
        "tune_threshold": cfg.threshold.tune_on_train,
        "shap": False,
        "shap_sample": 1000,
        "experiment_name": cfg.mlflow.experiment_name,
        "run_name": None,
        "author": cfg.meta.author,
        "dataset_version": cfg.meta.dataset_version,
        "n_estimators": cfg.model.n_estimators,
        "max_depth": cfg.model.max_depth,
        "min_samples_leaf": cfg.model.min_samples_leaf,
        "C": cfg.model.C,
        "max_iter": cfg.model.max_iter,
        "learning_rate": cfg.model.learning_rate,
    }

    if trial is not None:
        if model_name == "rf":
            params["n_estimators"] = trial.suggest_int("n_estimators", 100, 500)
            params["max_depth"] = trial.suggest_int("max_depth", 4, 20)
            params["min_samples_leaf"] = trial.suggest_int("min_samples_leaf", 1, 10)

        elif model_name == "lr":
            params["C"] = trial.suggest_float("C", 1e-3, 10.0, log=True)
            params["max_iter"] = trial.suggest_int("max_iter", 500, 2000)

        elif model_name == "xgb":
            params["n_estimators"] = trial.suggest_int("n_estimators", 100, 500)
            params["max_depth"] = trial.suggest_int("max_depth", 3, 10)
            params["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)

        elif model_name == "lgbm":
            params["n_estimators"] = trial.suggest_int("n_estimators", 100, 500)
            params["max_depth"] = trial.suggest_int("max_depth", 3, 16)
            params["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)

    return ArgsNamespace(**params)


def prepare_train_val_data(cfg: DictConfig):
    train_df = pd.read_csv(cfg.data.train_path)

    X = train_df.drop(columns=[cfg.data.target])
    y = train_df[cfg.data.target].astype(int)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=cfg.hpo.validation_size,
        random_state=SEED,
        stratify=y,
    )

    if cfg.preprocessing.scale_time_amount:
        X_train, X_val = apply_scaling_if_needed(X_train, X_val)

    if cfg.preprocessing.imbalance == "smote":
        X_train, y_train = apply_smote(X_train, y_train, random_state=SEED)

    return X_train, X_val, y_train, y_val


def objective_factory(cfg, X_train, X_val, y_train, y_val):
    def objective(trial):
        args = make_args_from_cfg(cfg, trial)
        model = build_model(args)

        with mlflow.start_run(nested=True):
            model.fit(X_train, y_train)

            proba_val = model.predict_proba(X_val)[:, 1]

            y_pred = (proba_val >= args.threshold).astype(int)
            metrics = compute_metrics(y_val, y_pred, proba_val)

            mlflow.log_params(trial.params)
            for k, v in metrics.items():
                mlflow.log_metric(f"{k}_val", float(v))

            return float(metrics[cfg.hpo.optimize_metric])

    return objective


def build_sampler(cfg):
    if cfg.hpo.sampler == "tpe":
        return optuna.samplers.TPESampler(seed=SEED)
    return optuna.samplers.RandomSampler(seed=SEED)


def retrain_and_log_best(cfg, best_params):
    train_df = pd.read_csv(cfg.data.train_path)
    test_df = pd.read_csv(cfg.data.test_path)

    X_train = train_df.drop(columns=[cfg.data.target])
    y_train = train_df[cfg.data.target].astype(int)

    X_test = test_df.drop(columns=[cfg.data.target])
    y_test = test_df[cfg.data.target].astype(int)

    args = make_args_from_cfg(cfg)

    for k, v in best_params.items():
        setattr(args, k, v)

    if args.scale_time_amount:
        X_train, X_test = apply_scaling_if_needed(X_train, X_test)

    if args.imbalance == "smote":
        X_train, y_train = apply_smote(X_train, y_train, random_state=SEED)

    model = build_model(args)
    model.fit(X_train, y_train)

    proba_test = model.predict_proba(X_test)[:, 1]
    y_pred = (proba_test >= args.threshold).astype(int)

    metrics = compute_metrics(y_test, y_pred, proba_test)

    os.makedirs("models_hpo", exist_ok=True)
    os.makedirs("reports_hpo", exist_ok=True)

    model_path = "models_hpo/best_model.pkl"
    joblib.dump(model, model_path)

    with open("reports_hpo/best_params.json", "w") as f:
        import json
        json.dump(best_params, f, indent=2)

    with open("reports_hpo/final_config_resolved.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    mlflow.log_artifact(model_path)
    mlflow.log_artifact("reports_hpo/best_params.json")
    mlflow.log_artifact("reports_hpo/final_config_resolved.yaml")
    mlflow.sklearn.log_model(model, "best_model")

    for k, v in metrics.items():
        mlflow.log_metric(f"{k}_test", float(v))

    print("Best params:", best_params)
    print("Test metrics:", metrics)


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    X_train, X_val, y_train, y_val = prepare_train_val_data(cfg)
    sampler = build_sampler(cfg)

    study = optuna.create_study(direction="maximize", sampler=sampler)

    with mlflow.start_run():
        study.optimize(
            objective_factory(cfg, X_train, X_val, y_train, y_val),
            n_trials=cfg.hpo.n_trials,
        )

        best_params = study.best_params
        mlflow.log_params(best_params)

        retrain_and_log_best(cfg, best_params)


if __name__ == "__main__":
    main()