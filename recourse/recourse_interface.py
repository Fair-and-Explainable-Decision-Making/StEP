from ..models.model_interface import ModelInterface
from ..data.data_interface import DataInterface
import pandas as pd
import abc


class RecourseInterface(abc.ABC):
    def __init__(self, model:ModelInterface, data_interface:DataInterface) -> None:
        pass

    @abc.abstractmethod
    def get_counterfactuals(self, poi: pd.DataFrame):
        pass

    @abc.abstractmethod
    def get_paths(self, poi: pd.DataFrame):
        pass