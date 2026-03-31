from airflow.models import DagBag


def test_dag_import():
    dag_bag = DagBag(dag_folder="dags/", include_examples=False)
    assert len(dag_bag.import_errors) == 0, f"DAG import errors: {dag_bag.import_errors}"


def test_dag_exists():
    dag_bag = DagBag(dag_folder="dags/", include_examples=False)
    assert "ml_training_pipeline" in dag_bag.dags