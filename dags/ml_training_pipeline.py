from datetime import datetime
import json
import os

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator


def check_data():
    if os.path.exists("/opt/airflow/project/data/ci/train_sample.csv") and os.path.exists("/opt/airflow/project/data/ci/test_sample.csv"):
        return True
    raise FileNotFoundError("CI sample datasets not found")


def read_metrics(**context):
    metrics_path = "/opt/airflow/project/artifacts/metrics.json"
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    context["ti"].xcom_push(key="metrics", value=metrics)
    return metrics


def choose_branch(**context):
    metrics = context["ti"].xcom_pull(task_ids="evaluate_model", key="metrics")
    if metrics["f1"] >= 0.5:
        return "register_model"
    return "stop_pipeline"


with DAG(
    dag_id="ml_training_pipeline",
    start_date=datetime(2025, 1, 1),
    schedule="@daily",
    catchup=False,
    tags=["mlops", "lab5"],
) as dag:

    check_data_task = PythonOperator(
        task_id="check_data",
        python_callable=check_data,
    )

    prepare_data = BashOperator(
        task_id="prepare_data",
        bash_command="cd /opt/airflow/project && dvc repro prepare || echo 'prepare skipped for CI sample mode'"
    )

    train_model = BashOperator(
        task_id="train_model",
        bash_command=(
            "cd /opt/airflow/project && "
            "python scripts/train_for_airflow.py "
            "--train_path data/ci/train_sample.csv "
            "--test_path data/ci/test_sample.csv "
            "--target Class "
            "--model rf "
            "--model_out artifacts/model.pkl "
            "--metrics_out artifacts/metrics.json"
        ),
    )

    evaluate_model = PythonOperator(
        task_id="evaluate_model",
        python_callable=read_metrics,
    )

    branch_task = BranchPythonOperator(
        task_id="branch_on_quality",
        python_callable=choose_branch,
    )

    register_model = BashOperator(
        task_id="register_model",
        bash_command="cd /opt/airflow/project && python scripts/register_model.py",
    )

    stop_pipeline = EmptyOperator(task_id="stop_pipeline")

    check_data_task >> prepare_data >> train_model >> evaluate_model >> branch_task
    branch_task >> register_model
    branch_task >> stop_pipeline