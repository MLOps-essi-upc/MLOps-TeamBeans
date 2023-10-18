import pytest
from src.models.train_model import CustomDataset
from src.models.train_model import SimpleCNNReducedStride10
from src.models.train_model import main
from src.data.make_dataset import CustomDataset

import mlflow
from unittest.mock import patch

from torch.utils.data import Dataset
import torch
import os
import mlflow.pytorch
from mlflow import MlflowClient
from dotenv import load_dotenv
import torch.nn as nn
from codecarbon import EmissionsTracker
from torchvision import transforms


@patch('builtins.input', side_effect=['2'])

def test_training_loss_decreasing(mock_input):
    # Run the training process
    main()

    # Fetch the logged epoch loss metric from MLflow
    with mlflow.start_run():
        run = mlflow.search_runs(filter_string="").iloc[0]
        epoch_losses = []
        for e in range(5):
            loss_key = "individual_epoch_loss_" + str(e+1)
            loss_value = run.data.metrics.get(loss_key)
            if loss_value is not None:
                epoch_losses.append(loss_value)

    # Check if the individual epoch loss is decreasing
    assert len(epoch_losses) > 1  # Ensure we have more than one value
    assert all(epoch_losses[i] >= epoch_losses[i + 1] for i in range(len(epoch_losses) - 1))





### To be added: check if the final accuracy is higher than 70%.

# import pytest
# from unittest.mock import patch
# from src.models.train_model import main
# import mlflow

# @patch('builtins.input', side_effect=['1', '2'])
# def test_training_loss_decreasing(mock_input):
#     # Run the training process
#     main()

#     # Fetch the logged epoch loss metric from MLflow
#     with mlflow.start_run():
#         run = mlflow.search_runs(filter_string="").iloc[0]
#         epoch_losses = []
#         for e in range(5):
#             loss_key = "individual_epoch_loss_" + str(e+1)
#             loss_value = run.data.metrics.get(loss_key)
#             if loss_value is not None:
#                 epoch_losses.append(loss_value)

#     # Check if the individual epoch loss is decreasing
#     assert len(epoch_losses) > 1  # Ensure we have more than one value
#     assert all(epoch_losses[i] >= epoch_losses[i + 1] for i in range(len(epoch_losses) - 1))
