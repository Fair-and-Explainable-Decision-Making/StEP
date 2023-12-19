import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data as data
import numpy as np
import pandas as pd

from models.pytorch_models.dnn_basic import BaselineDNN
from models.pytorch_models.logreg import LogisticRegression


class PyTorchModel:
    """
    TODO: docstring
    Indicate to the user that they can use any custom PyTorch architecture that inherits from nn.Module
    """
    def __init__(self, model, criterion = nn.BCELoss(), lr = 1e-4, weight_decay = 1e-4, epochs = 5, batch_size=1):
        self._model = model
        self._criterion = criterion
        self._lr = lr
        self._weight_decay = weight_decay
        self._epochs = epochs
        self._batch_size = batch_size
        

    def fit(self, features, labels):
        if isinstance(features, pd.DataFrame):
            features = features.to_numpy()
        if isinstance(labels, pd.Series):
            labels = labels.to_numpy()
        if not isinstance(self._model, nn.Module):
            raise Exception("Not a compatible PyTorch implementation.")
        tensor_features = torch.Tensor(features)
        tensor_labels = torch.Tensor(labels.flatten())
        train_data = torch.utils.data.TensorDataset(tensor_features,tensor_labels)
        trainloader = torch.utils.data.DataLoader(train_data, batch_size=self._batch_size)

        model = self._model
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        optimizer = optim.SGD(model.parameters(), lr=self._lr, weight_decay = self._weight_decay)

        for epoch in range(self._epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, train_data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, target]
                inputs, target = train_data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)
                loss = self._criterion(outputs, target.unsqueeze(-1))
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 1000 == 999:    # print every 1000 mini-batches
                    print(i)
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 1000))
                    running_loss = 0.0

        #print('Finished Training')
        self._model = model
        return self._model

    def predict_proba(self, features, grad=False):
        if isinstance(features, pd.DataFrame) or isinstance(features, pd.Series):
            features = features.to_numpy()
        tensor_features = torch.Tensor(features)
        if grad:
            outputs = self._model(tensor_features)
        else: 
            with torch.no_grad():
                outputs = self._model(tensor_features)
        return outputs.numpy()
    
    def predict(self, features):
        out = self.predict_proba(features)[:, 0]
        return np.array(out > 0.5, dtype=np.int32)
