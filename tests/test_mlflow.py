import pytest
import mlflow
from dotenv import load_dotenv
import os


def test_mlflow_connection():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    load_dotenv(dotenv_path=os.path.join(project_root,'.env'))
    assert os.getenv('MLFLOW_TRACKING_USERNAME') is not None, "Warning: MLFLOW_TRACKING_USERNAME is not defined"
    assert os.getenv('MLFLOW_TRACKING_PASSWORD') is not None, "Warning: MLFLOW_TRACKING_PASSWORD is not defined"

    os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv('MLFLOW_TRACKING_USERNAME')
    os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv('MLFLOW_TRACKING_PASSWORD')
    try:
        mlflow.set_tracking_uri('https://dagshub.com/wwoszczek/MLOps-TeamBeans.mlflow')
    except Exception:
        pytest.fail("MLflow API connection failed")
    

    runs = mlflow.search_runs(experiment_names=["CNN-pytorch"])
    n=5 #number of runs to be tested
    runs_final=runs.head(n)
    assert len(runs_final['run_id'])== n, "MLflow API connection needs to be checked."