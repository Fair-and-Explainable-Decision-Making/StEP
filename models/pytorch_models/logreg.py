import torch
import torch.nn as nn


class LogisticRegression(nn.Module):
    """
    Logisitc Regression model using PyTorch.
    Parameters
    ----------
    num_features : int
        Number of features of the dataset that will be passed in.
    num_classes : int
        Number of unque labels/classes for the data. 
    """

    def __init__(self, num_features: int, num_classes: int = 2):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(num_features, num_classes)

    def forward(self, features):
        return torch.sigmoid(self.linear(features))
