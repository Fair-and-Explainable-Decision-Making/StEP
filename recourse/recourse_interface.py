from ..models.model_interface import ModelInterface
from ..data.data_interface import DataInterface
import pandas as pd
import abc


class RecourseInterface(abc.ABC):
    def __init__(self, model:ModelInterface, data_interface:DataInterface, backend:str = "sklearn") -> None:
        pass

    @abc.abstractmethod
    def get_counterfactuals(self, X:pd.DataFrame, num_CFs:int = 1, sparsity_param:float = 0.1):
        pass

    @abc.abstractmethod
    def get_path(self):
        pass