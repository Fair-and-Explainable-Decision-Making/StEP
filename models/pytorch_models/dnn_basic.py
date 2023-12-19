import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data as data


class BaselineDNN(nn.Module):

    def __init__(self, input_size: int, num_classes: int = 1):
        super(BaselineDNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 16)
        self.fc2 = nn.Linear(16, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = F.sigmoid(out)
        return out