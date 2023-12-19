import pandas as pd
from typing import Union
import sklearn 
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from models.pytorch_wrapper import PyTorchModel
from models.pytorch_models.dnn_basic import BaselineDNN
from models.pytorch_models.logreg import LogisticRegression as LogReg

class ModelInterface():
    def __init__(self, model: Union[sklearn.base.BaseEstimator, PyTorchModel]):
        self._model = model
        if not (isinstance(self._model, sklearn.base.BaseEstimator) or isinstance(self._model, PyTorchModel)):
            raise Exception("Invalid model backend")

    def fit(self, X: Union[np.ndarray, pd.DataFrame, torch.Tensor],
                y:Union[np.ndarray, pd.DataFrame, torch.Tensor]) -> np.ndarray:
        if isinstance(self._model, sklearn.base.BaseEstimator):
            y=np.ravel(y)
        self._model.fit(X,y)
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame, torch.Tensor]) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame, torch.Tensor]) -> np.ndarray:
        pred = self._model.predict_proba(X)
        if isinstance(self._model, sklearn.base.BaseEstimator):
            return np.array(pred[:,1]).reshape(-1,1)
        elif isinstance(self._model, PyTorchModel):
            return pred

    def get_model(self) -> str:
        return self._model
    