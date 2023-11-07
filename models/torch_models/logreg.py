import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim


class LogisticRegression(nn.Module):
    """
    Logisitc Regression model using PyTorch
    Args:
        input_dim: num of features of the dataset that will be passed in
    """
    def __init__(self, input_dim, n_classes=1):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, n_classes)

    def forward(self, X):
        return torch.sigmoid(self.linear(X))
