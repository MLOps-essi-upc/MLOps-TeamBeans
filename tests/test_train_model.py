import mlflow
import pytest
import os
from dotenv import load_dotenv


project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(dotenv_path=os.path.join(project_root,'.env'))
os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv('MLFLOW_TRACKING_USERNAME')
os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv('MLFLOW_TRACKING_PASSWORD')

mlflow.set_tracking_uri('https://dagshub.com/wwoszczek/MLOps-TeamBeans.mlflow')

runs = mlflow.search_runs(experiment_names=["CNN-pytorch"])
runs_sort=runs.sort_values(by='start_time',ascending=False)
# print(runs_sort)

n=1 #number of runs to be tested

runs_final=runs_sort.head(n)

def test_epoch_loss():
    loss_values=[]
    for index, row in runs_final.iterrows():
        for i in range(5):
            loss_key = "metrics.individual_epoch_loss_" + str(i+1)
            loss_value = row[loss_key]
            assert loss_value is not None, "Missing metrics for individual epoch loss"
            loss_values.append(loss_value)

        # Check if the individual epoch loss is decreasing
        assert all(loss_values[i] > loss_values[i + 1] for i in range(len(loss_values) - 1))

def test_accuracy_threshold():
    for index, row in runs_final.iterrows():
        assert row["metrics.validation_accuracy"]>70, "Model accuracy under threshold, review model architecture"

def test_artifacts():
    for index, row in runs_final.iterrows():
        assert row["artifact_uri"] is not None,"Model not stored as artifact"

