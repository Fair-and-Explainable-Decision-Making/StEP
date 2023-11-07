import pandas as pd
from typing import Sequence, Any, Tuple, Union
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import sklearn 
import numpy as np
import torch
import torch.nn as nn

#from  data.synthetic_data import create_synthetic_data
#from  data.data_interface import DataInterface
from torch_wrapper import TorchModel
from sklearn.linear_model import LogisticRegression

class ModelInterface():
    def __init__(self, model : Union[sklearn.base.BaseEstimator, TorchModel]):
        self._model = model
        if not (isinstance(self._model, sklearn.base.BaseEstimator) or isinstance(self._model, TorchModel)):
            raise Exception("Invalid model backend")

    def fit(self, X:Union[np.ndarray, pd.DataFrame, torch.Tensor],
                y:Union[np.ndarray, pd.DataFrame, torch.Tensor]) -> np.ndarray:
        if isinstance(self._model, sklearn.base.BaseEstimator):
            y=np.ravel(y)
        self._model.fit(X,y)
    
    def predict(self, X : Union[np.ndarray, pd.DataFrame, torch.Tensor]) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X : Union[np.ndarray, pd.DataFrame, torch.Tensor]) -> np.ndarray:
        pred = self._model.predict_proba(X)
        if isinstance(self._model, sklearn.base.BaseEstimator):
            return np.array(pred[:,1]).reshape(-1,1)
        elif isinstance(self._model, TorchModel):
            return pred

    def get_model(self) -> str:
        return self._model
    
if __name__ == "__main__":
    X = np.random.random((1000, 3))
    Z = np.random.randint(2, size=1000).reshape(-1,1)

    XZ = np.hstack([X,Z])
    w = np.array([1.0, -2.0, 3.0, 2.0])
    y = 1/(1 + np.exp(-XZ @ w))
    Y = np.array(y >= 0.5, dtype=np.int32).reshape(-1,1)
    print(Y[:10])
    model = TorchModel(model_type="dnn",batch_size=1)
    mi = ModelInterface(model)
    mi.fit(XZ,Y)
    print("DNN")
    print(mi.predict(XZ)[:10])
    print(mi.predict_proba(XZ)[:10])
    print()

    model = TorchModel(model_type="logreg",batch_size=1)
    mi = ModelInterface(model)
    mi.fit(XZ,Y)
    print("Log Reg PyTorch")
    print(mi.predict(XZ)[:10])
    print(mi.predict_proba(XZ)[:10])
    print()

    print("Log Reg Sklearn")
    model = LogisticRegression()
    mi = ModelInterface(model)
    mi.fit(XZ,Y)
    print(mi.predict(XZ)[:10])
    print(mi.predict_proba(XZ)[:10])