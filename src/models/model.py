import torch
import torch.nn.functional as F
import torch.nn as nn

class SimpleCNNReducedStride10(nn.Module):
    """
    This is the architecture of our CNN.
    """
    def __init__(self, num_classes=3):
        super(SimpleCNNReducedStride10, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=2, padding=1)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.relu2 = nn.ReLU()
        
        self.dropout = nn.Dropout(0.5)  # Add dropout for regularization
        
        # Calculate the correct input size for fc1 based on the spatial dimensions
        self.fc1_input_size = self.calculate_fc1_input_size()
        self.fc1 = nn.Linear(250000, 256)
        self.relu3 = nn.ReLU()
        
        self.dropout2 = nn.Dropout(0.5)  # Add dropout for regularization
        
        self.fc2 = nn.Linear(256, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=1)  # Softmax activation for classification

    def calculate_fc1_input_size(self):
        # Assuming the output size after the second convolutional layer
        # with stride 10 is (16, 50, 50), calculate the input size for fc1
        return 16 * 50 * 50

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        
        x = x.view(x.size(0), -1)  # Flatten the feature maps
        x = self.dropout(x)  # Apply dropout for regularization
        
        x = self.fc1(x)
        
        x = self.relu3(x)
        x = self.dropout2(x)
        
        x = self.fc2(x)
        
        x = self.log_softmax(x)  # Apply softmax for classification
        return x