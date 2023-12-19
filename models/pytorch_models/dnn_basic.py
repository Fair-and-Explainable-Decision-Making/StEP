import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data as data


class BaselineDNN(nn.Module):
    """
    TODO: docstring
    """
    def __init__(self, num_features: int, num_classes: int = 1):
        super(BaselineDNN, self).__init__()
        self.fc1 = nn.Linear(num_features, 16)
        self.fc2 = nn.Linear(16, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, features):
        out = self.fc1(features)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = F.sigmoid(out)
        return out