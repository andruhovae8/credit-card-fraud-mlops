import json
import sys


def main():
    metrics_path = "artifacts/metrics.json"
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    print(json.dumps(metrics))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())