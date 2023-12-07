import pandas as pd
from typing import Union
import sklearn 
import numpy as np
import torch

#from  data.synthetic_data import create_synthetic_data
#from  data.data_interface import DataInterface
from pytorch_wrapper import PyTorchModel
from sklearn.linear_model import LogisticRegression
from pytorch_models.dnn_basic import DNN
from pytorch_models.logreg import LogisticRegression as LogReg

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
    
if __name__ == "__main__":
    X = np.random.random((1000, 3))
    Z = np.random.randint(2, size=1000).reshape(-1,1)

    XZ = np.hstack([X,Z])
    w = np.array([1.0, -2.0, 3.0, 2.0])
    y = 1/(1 + np.exp(-XZ @ w))
    Y = np.array(y >= 0.5, dtype=np.int32).reshape(-1,1)
    XZ = pd.DataFrame(XZ, columns=['X1','X2','X3','Z'])
    Y = pd.Series(Y.flatten())
    print(XZ)
    print(Y[:10])
    model = PyTorchModel(DNN(XZ.shape[1]),batch_size=1)
    mi = ModelInterface(model)
    mi.fit(XZ,Y)
    print("DNN")
    print(mi.predict(XZ)[:10])
    print(mi.predict_proba(XZ)[:10])
    print()
    XZ_test = XZ.sample(frac=1)
    probs = mi.predict_proba(XZ_test)
    conf_df = pd.Series(probs.flatten(), index=XZ_test.index)
    print(XZ_test.head(5))
    print(conf_df.head(5))
    conf_inter = 0.55
    conf_df = conf_df[conf_df >= conf_inter]
    print(conf_df)