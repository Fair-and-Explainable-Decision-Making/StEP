import pandas as pd
from typing import Union
import sklearn 
import numpy as np
import torch

from models.pytorch_wrapper import PyTorchModel

DATA_TYPES= Union[np.ndarray, pd.DataFrame, torch.Tensor]

class ModelInterface():
    """
    Creates a model interface based on a sklearn BaseEstimator such as LogisticRegression or
    on a PyTorch models that has been wrapped by PyTorchModel.

    Parameters
    ----------
    model : Union[sklearn.base.BaseEstimator, PyTorchModel]
        model to be used
    """
    def __init__(self, model: Union[sklearn.base.BaseEstimator, PyTorchModel]):
        self._model = model
        if not (isinstance(self._model, sklearn.base.BaseEstimator) 
                or isinstance(self._model, PyTorchModel)):
            raise Exception("Invalid model backend")

    def fit(self, features: DATA_TYPES, labels: DATA_TYPES) -> np.ndarray:
        """
        Trains the model with the passed in data.

        Parameters
        ----------
        features : Union[np.ndarray, pd.DataFrame, torch.Tensor]
            A tabular data object with the features and samples used for training.
        labels : Union[np.ndarray, pd.DataFrame, torch.Tensor]
            A tabular data object with the labels for each sample used for training.
        """
        if isinstance(self._model, sklearn.base.BaseEstimator):
            labels = np.ravel(labels)
        self._model.fit(features, labels)
    
    def predict(self, features: DATA_TYPES) -> np.ndarray:
        """
        Uses model's predict function to make binary decisions. Threshold for binary decision
        is 0.50.

        Parameters
        ----------
        features : Union[np.ndarray, pd.DataFrame, torch.Tensor]
            A tabular data object with the features and samples used for inference.
        Returns
        -------
        model.predict(features) : np.ndarray
            Array of binary outcomes.
        """
        return self._model.predict(features)

    def predict_proba(self, features: DATA_TYPES) -> np.ndarray:
        """
        Uses model's predict probability function for the probability estimate
        for the positive class.

        Parameters
        ----------
        features : Union[np.ndarray, pd.DataFrame, torch.Tensor]
            A tabular data object with the features and samples used for inference.
        Returns
        -------
        model.predict_proba(features) : np.ndarray
            Array of positive probability estimates.
        """
        pred = self._model.predict_proba(features)
        if isinstance(self._model, sklearn.base.BaseEstimator):
            return np.array(pred[:,1]).reshape(-1,1)
        elif isinstance(self._model, PyTorchModel):
            return pred

    def get_model(self) -> str:
        """
        Returns the model used by the interface.
        """
        return self._model
    