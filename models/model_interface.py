import pandas as pd
from typing import Union
import sklearn 
import numpy as np
import torch

from models.pytorch_wrapper import PyTorchModel

DATA_TYPES= Union[np.ndarray, pd.DataFrame, torch.Tensor]

class ModelInterface():
    """
    TODO: docstring
    """
    def __init__(self, model: Union[sklearn.base.BaseEstimator, PyTorchModel]):
        self._model = model
        if not (isinstance(self._model, sklearn.base.BaseEstimator) 
                or isinstance(self._model, PyTorchModel)):
            raise Exception("Invalid model backend")

    def fit(self, features: DATA_TYPES, labels: DATA_TYPES) -> np.ndarray:
        if isinstance(self._model, sklearn.base.BaseEstimator):
            labels = np.ravel(labels)
        self._model.fit(features, labels)
    
    def predict(self, features: DATA_TYPES) -> np.ndarray:
        return self._model.predict(features)

    def predict_proba(self, features: DATA_TYPES) -> np.ndarray:
        pred = self._model.predict_proba(features)
        if isinstance(self._model, sklearn.base.BaseEstimator):
            return np.array(pred[:,1]).reshape(-1,1)
        elif isinstance(self._model, PyTorchModel):
            return pred

    def get_model(self) -> str:
        return self._model
    