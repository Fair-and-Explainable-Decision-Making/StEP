import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data as data


class BaselineDNN(nn.Module):
    """
    Baseline deep neural network using PyTorch.
    Parameters
    ----------
    num_features : int
        Number of features of the dataset that will be passed in.
    num_classes : int
        Number of unque labels/classes for the data. 
    """

    def __init__(self, num_features: int, num_classes: int = 1):
        super(BaselineDNN, self).__init__()
        self.fc1 = nn.Linear(num_features, 16)
        self.fc2 = nn.Linear(16, 32)
        self.fc3 = nn.Linear(32, 2)
        self.relu = nn.ReLU()

    def forward(self, features):
        out = self.fc1(features)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = F.sigmoid(out)
        return out
