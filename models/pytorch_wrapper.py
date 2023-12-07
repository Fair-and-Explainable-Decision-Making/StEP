import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data as data

import numpy as np
from pytorch_models.dnn_basic import DNN
from pytorch_models.logreg import LogisticRegression
import pandas as pd


class PyTorchModel:
    def __init__(self, model, criterion = nn.BCELoss(), lr = 1e-4, weight_decay = 1e-4, epochs = 5, batch_size=1):
        self._xtrain = None
        self._ytrain = None
        self._model = model
        self._criterion = criterion
        self._lr = lr
        self._weight_decay = weight_decay
        self._epochs = epochs
        self._batch_size = batch_size
        

    def fit(self, xtrain, ytrain):
        self._xtrain = xtrain
        self._ytrain = ytrain
        if isinstance(self._xtrain, pd.DataFrame):
            self._xtrain = self._xtrain.to_numpy()
        if isinstance(self._ytrain, pd.Series):
            self._ytrain = self._ytrain.to_numpy()
        if not isinstance(self._model, nn.Module):
            raise Exception("Not a compatible PyTorch implementation.")
        tensor_x = torch.Tensor(self._xtrain)
        tensor_y = torch.Tensor(self._ytrain.flatten())
        train_data = torch.utils.data.TensorDataset(tensor_x,tensor_y)
        trainloader = torch.utils.data.DataLoader(train_data, batch_size=self._batch_size)

        model = self._model
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        criterion = self._criterion
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
                loss = criterion(outputs, target.unsqueeze(-1))
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

    def predict_proba(self, X, grad=False):
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X = X.to_numpy()
        tensor_x = torch.Tensor(X)
        if grad:
            outputs = self._model(tensor_x)
        else: 
            with torch.no_grad():
                outputs = self._model(tensor_x)
        return outputs.numpy()
    
    def predict(self, X):
        out = self.predict_proba(X)[:, 0]
        return np.array(out > 0.5, dtype=np.int32)


if __name__ == "__main__":
    X = np.random.random((1000, 3))
    Z = np.random.randint(2, size=1000).reshape(-1,1)

    XZ = np.hstack([X,Z])
    w = np.array([1.0, -2.0, 3.0, 2.0])
    y = 1/(1 + np.exp(-XZ @ w))
    Y = np.array(y >= 0.5, dtype=np.int32).reshape(-1,1)
    print(Y[:10])
    model = PyTorchModel(DNN(XZ.shape[1]),batch_size=1)
    model.fit(XZ,y)
    print(model.predict(XZ)[:10])
    print(model.predict_proba(XZ)[:10])