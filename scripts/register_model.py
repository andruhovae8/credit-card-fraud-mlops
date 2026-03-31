import json
import os
import shutil


def main():
    metrics_path = "artifacts/metrics.json"
    model_path = "artifacts/model.pkl"

    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    os.makedirs("registry/staging", exist_ok=True)

    target_model_path = "registry/staging/model.pkl"
    target_metrics_path = "registry/staging/metrics.json"

    shutil.copy2(model_path, target_model_path)

    with open(target_metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("Model registered to local staging registry")
    print(f"Saved model to: {target_model_path}")
    print(f"Saved metrics to: {target_metrics_path}")


if __name__ == "__main__":
    main()