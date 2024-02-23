import torch
import torch.nn as nn
import torch.optim as optim
from typing import Union
from torch.utils import data as data
import numpy as np
import pandas as pd

DATA_TYPES = Union[np.ndarray, pd.DataFrame, torch.Tensor]


class PyTorchModel:
    """
    Wrapper for a PyTorch models that inherit from nn.Module. 
    PyTorch model must have constructor with architecture and a forward function.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model to be used. 
    criterion : torch.nn loss function
        Criterion that measures the loss between the target and the input probabilities.
    lr : float
        Learning rate.
    weight_decay : float
        Weight decay.
    epochs : int
        Number of epochs for training.
    batch_size : int
        Number of samples for each batch.
    """

    def __init__(self, model, validation_features=None, validation_labels=None, criterion=nn.BCELoss(), lr=1e-4, weight_decay=1e-4, epochs=5, batch_size=1):
        self._model = model
        self._criterion = criterion
        self._lr = lr
        self._weight_decay = weight_decay
        self._epochs = epochs
        self._batch_size = batch_size

        if validation_features is not None and validation_labels is not None:
            if isinstance(validation_features, pd.DataFrame):
                validation_features = validation_features.to_numpy()
            if isinstance(validation_labels, pd.Series):
                validation_labels = validation_labels.to_numpy()
            tensor_features = torch.Tensor(validation_features)
            tensor_labels = torch.Tensor(validation_labels.flatten())
            valid_data = torch.utils.data.TensorDataset(
                tensor_features, tensor_labels)
            self._validloader = torch.utils.data.DataLoader(
                valid_data, batch_size=self._batch_size)

    def fit(self, features, labels):
        """
        Trains the PyTorch model with the passed in data and parameters defined in the constuctor.

        Parameters
        ----------
        features : Union[np.ndarray, pd.DataFrame, torch.Tensor]
            A tabular data object with the features and samples used for training.
        labels : Union[np.ndarray, pd.DataFrame, torch.Tensor]
            A tabular data object with the labels for each sample used for training.
        """
        if isinstance(features, pd.DataFrame):
            features = features.to_numpy()
        if isinstance(labels, pd.Series):
            labels = labels.to_numpy()
        if not isinstance(self._model, nn.Module):
            raise Exception("Not a compatible PyTorch implementation.")

        
        _, counts = np.unique(labels, return_counts=True)
        class_weights = torch.FloatTensor(len(labels)/ (counts * 2))
        
        self._criterion = nn.BCELoss(weight=class_weights)
        tensor_features = torch.Tensor(features)
        tensor_labels = torch.Tensor(labels.flatten())
        train_data = torch.utils.data.TensorDataset(
            tensor_features, tensor_labels)
        trainloader = torch.utils.data.DataLoader(
            train_data, batch_size=self._batch_size)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = self._model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=self._lr)

        for epoch in range(self._epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, train_data in enumerate(trainloader):
                # get the inputs; data is a list of [inputs, target]
                inputs, target = train_data
                inputs = inputs.to(device)
                target = torch.nn.functional.one_hot(
                    target.to(device).type(torch.int64), num_classes=2)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)
                
                loss = self._criterion(outputs, target.float())
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            running_vloss = 0.0
            model.eval()
            with torch.no_grad():
                for i, valid_data in enumerate(self._validloader):
                    vinputs, vtarget = valid_data
                    vinputs = inputs.to(device)
                    vtarget = torch.nn.functional.one_hot(
                        vtarget.to(device).type(torch.int64), num_classes=2)
                    voutputs = model(vinputs)
                    vloss = self._criterion(voutputs, vtarget.float())
                    running_vloss += vloss
            print('EPOCH {} LOSS train {} valid {}'.format(
                epoch, running_loss/len(trainloader), running_vloss/len(self._validloader)))
        self._model = model
        return self._model

    def predict_proba(self, features, grad=False):
        """
        Runs a fowards pass for inference to get the probability estimate
        for each label.

        Parameters
        ----------
        features : Union[np.ndarray, pd.DataFrame, torch.Tensor]
            A tabular data object with the features and samples used for inference.
        Returns
        -------
        outputs : np.ndarray
            Array of probability estimates for each label.
        """
        if isinstance(features, pd.DataFrame) or isinstance(features, pd.Series):
            features = features.to_numpy()
        if not isinstance(features, torch.Tensor):
            features = torch.Tensor(features.astype(float))
        if grad:
            outputs = self._model(features)
        else:
            with torch.no_grad():
                outputs = self._model(features)
        return outputs.numpy()

    def predict(self, features):
        """
        Makes binary predictions based on input data. Threshold for binary decision
        is 0.50.

        Parameters
        ----------
        features : np.ndarray
            A tabular data object with the features and samples used for inference.
        Returns
        -------
        (out > 0.5) : np.ndarray
            Array of binary outcomes.
        """
        out = self.predict_proba(features)[:, 1]
        return np.array(out >= 0.5, dtype=np.int32)

    def get_model(self):
        """
        Returns the model used by the interface.
        """
        return self._model
    
    def save_model(self, file_path):
        torch.save(self._model.state_dict(), file_path)

    def load_model(self, file_path):
        self._model.load_state_dict(torch.load(file_path))
