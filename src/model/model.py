import torch
import torch.nn as nn
import torch.nn.functional as F
from . import config

class PilotNetPyTorch(nn.Module):
    def __init__(self, input_channels=3):
        super(PilotNetPyTorch, self).__init__()
        # Convolutional layers
        # Input shape: (N, C, H, W) -> (N, input_channels, 66, 200)
        self.conv1 = nn.Conv2d(input_channels, 24, kernel_size=5, stride=2) # Out: (N, 24, 31, 98)
        self.conv2 = nn.Conv2d(24, 36, kernel_size=5, stride=2)           # Out: (N, 36, 14, 47)
        self.conv3 = nn.Conv2d(36, 48, kernel_size=5, stride=2)           # Out: (N, 48, 5, 22)
        self.conv4 = nn.Conv2d(48, 64, kernel_size=3, stride=1)           # Out: (N, 64, 3, 20)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1)           # Out: (N, 64, 1, 18)

        # Calculate the flattened size based on the output of the last conv layer
        # For input H=66, W=200, the output of conv5 is (N, 64, 1, 18)
        self.flattened_size = 64 * 1 * 18

        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, 1) # Output layer for steering angle

    def forward(self, x):
        # Convolutional block
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))

        # Flatten
        x = x.view(-1, self.flattened_size) # Or x.view(x.size(0), -1)

        # Fully connected block
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x) # No activation for regression output
        return x

def build_pilotnet_model(input_channels=config.IMG_CHANNELS):
    """
    Builds the PilotNet model using PyTorch.
    """
    model = PilotNetPyTorch(input_channels=input_channels)
    return model
