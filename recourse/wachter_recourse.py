from models import model_interface
from data import data_interface
import dice_ml
import pandas as pd
from recourse.recourse_interface import RecourseInterface
from typing import Optional, Sequence
from raiutils.exceptions import UserConfigValidationException
from scipy.stats import median_abs_deviation
import numpy as np

def wachter_objective(model, x_cf, x, lmbda, target, mad=None):
    return np.sum(np.abs(x_cf - x)/mad)

    if mad is None:
        return lmbda * ((model.predict(x_cf) - target) ** 2) + np.linalg.norm(x_cf - x, ord=2)**2
    else: 
        return lmbda * ((model.predict(x_cf) - target) ** 2) + np.sum(np.linalg.norm(x_cf - x, ord=1)/mad)
    
# Sparser Wachter
def sparse_wachter_objective(model, x_cf, x, lmbda, target):
    return np.linalg.norm(x_cf - x, ord=1) + np.linalg.norm(x_cf - x, ord=2) ** 2
    return lmbda * (model.predict(x_cf) - target) ** 2 + (np.linalg.norm(x_cf - x, ord=1) + np.linalg.norm(x_cf - x, ord=2) ** 2)

def mean_abs_dev(data):
    mads = []
    for c in range(data.shape[1]):
        mad_c = median_abs_deviation(data[:,c], scale='normal')
        if mad_c == 0:
            mads.append(1)
        else:
            mads.append(mad_c)
    mad = np.array(mads)
    return mad
def agnostic_wachter(x,target, model, data, lmbda=1, objective = "paper",n_samples = 10000, random_state = 0):
    x=x.values[0]
    data = data.sample(n=n_samples, random_state=random_state).values
    if objective == "paper":

        mad = mean_abs_dev(data)
        min_obj = wachter_objective(model, data[0], x, lmbda, target, mad)
        min_x_cf = data[0].reshape(1, -1)[0]

        for x_cf in data:
            obj = wachter_objective(model, x_cf, x, lmbda, target, mad)     
            if obj < min_obj:
                min_obj = obj
                min_x_cf = x_cf  
                
    elif objective == "euclidean":
        min_obj = wachter_objective(model, data[0], x, lmbda, target, mad)
        min_x_cf = data[0].reshape(1, -1)[0]

        for x_cf in data:
            obj = wachter_objective(model, x_cf, x, lmbda, target, mad)     
            if obj < min_obj:
                min_obj = obj
                min_x_cf = x_cf  
    elif objective =="sparse":
        min_obj = sparse_wachter_objective(model, data[0], x, lmbda, target)
        min_x_cf = data[0].reshape(1, -1)[0]

        for x_cf in data:
            obj = sparse_wachter_objective(model, x_cf, x, lmbda, target)     
            if obj < min_obj:
                min_obj = obj
                min_x_cf = x_cf  
    else:
        return None
    
    return min_x_cf

class WachterRecourse(RecourseInterface):
    def __init__(self, model: model_interface.ModelInterface, data_interface: data_interface.DataInterface, 
                 objective = "paper", confidence_threshold: float = 0.5,
                 random_seed: Optional[int] = None) -> None:
        """
        TODO: docstring
        """
        self._model = model
        self._data_interface = data_interface
        features, _, labels, _ = data_interface.get_train_test_split()

        positive_data = features.loc[labels[labels == 1].index]
        probs = self._model.predict_proba(
            positive_data.values, pos_label_only=True)
        positive_confident_df = pd.Series(
            probs.flatten(), index=positive_data.index)
        positive_confident_df = positive_confident_df[positive_confident_df >=
                                                        confidence_threshold]
        
        self._positive_data =  features.loc[positive_confident_df.index]
        self._objective = objective
        
        self._confidence_threshold = confidence_threshold
        self._random_seed = random_seed

    def get_counterfactuals(self, poi: pd.DataFrame):
        cf = agnostic_wachter(poi,1, self._model, self._positive_data, lmbda=1, objective = self._objective,n_samples = 1000, random_state = self._random_seed)   
        df = pd.DataFrame([cf], columns=self._positive_data.columns)
        return [df]
    def get_paths(self, poi: pd.DataFrame):
        
        cfs = self.get_counterfactuals(poi)
        paths = []
        for cf in cfs:
            paths.append([poi, cf])
        return paths

    def get_counterfactuals_from_paths(self, paths):
        # TODO: write docstring.
        cfs = []
        for p in paths:
            cfs.append(p[-1])
        return cfs
    