import pandas as pd
from typing import Union
import sklearn
import numpy as np
import torch

from models.pytorch_wrapper import PyTorchModel
from joblib import dump, load

DATA_TYPES = Union[np.ndarray, pd.DataFrame, torch.Tensor]


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

    def predict(self, features: DATA_TYPES, confidence_threshold = 0.5) -> np.ndarray:
        """
        Uses model's predict/predictproba function to make binary decisions. Threshold for binary decision
        is 0.50 by default.

        Parameters
        ----------
        features : Union[np.ndarray, pd.DataFrame, torch.Tensor]
            A tabular data object with the features and samples used for inference.
        Returns
        -------
        model.predict(features) : np.ndarray
            Array of binary outcomes.
        """
        if confidence_threshold == 0.5:
            return self._model.predict(features)
        out = self.predict_proba(features)[:, 1]
        return np.array(out >= confidence_threshold, dtype=np.int32)

    def predict_proba(self, features: DATA_TYPES, pos_label_only: bool = False) -> np.ndarray:
        """
        Uses model's predict probability function for the probability estimate
        for the positive class.

        Parameters
        ----------
        features : Union[np.ndarray, pd.DataFrame, torch.Tensor]
            A tabular data object with the features and samples used for inference.
        pos_label_only: bool
            True if you want a n x 1 probability estimates for positive class only for binary.
        Returns
        -------
        model.predict_proba(features) : np.ndarray
            Array of probability estimates.
        """
        pred = self._model.predict_proba(features)
        if pos_label_only:
            return np.array(pred[:, 1]).reshape(-1, 1)
        return pred

    def get_model(self):
        """
        Returns the model used by the interface.
        """
        if isinstance(self._model, sklearn.base.BaseEstimator):
            return self._model
        elif isinstance(self._model, PyTorchModel):
            return self._model.get_model()
        
    def save_model(self, file_path):
        if isinstance(self._model, sklearn.base.BaseEstimator):
            return dump(self._model, file_path+'.joblib')
        elif isinstance(self._model, PyTorchModel):
            return self._model.save_model(file_path+'.pt')
    
    def load_model(self, file_path):
        if isinstance(self._model, sklearn.base.BaseEstimator):
            self._model = load(self._model, file_path+'.joblib')
        elif isinstance(self._model, PyTorchModel):
            self._model.load_model(file_path+'.pt')
