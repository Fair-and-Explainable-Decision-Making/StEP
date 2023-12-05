from ..models.model_interface import ModelInterface
from ..data.data_interface import DataInterface
import pandas as pd
from recourse_interface import RecourseInterface
from step_lib import StEP
from typing import Optional


class StEPRecourse(RecourseInterface):
    def __init__(self, model: ModelInterface, data_interface:DataInterface,
        k_directions: int, use_train_data: bool = True,
        confidence_threshold: Optional[float] = None, random_seed: Optional[int] = None
    ) -> None:
        
        self.StEP_instance = StEP(k_directions, data_interface, model, use_train_data,
                                  confidence_threshold, random_seed)
    
    def get_counterfactuals(self, X: pd.DataFrame, num_CFs: int = 1, sparsity_param: float = 0.1):
        pass

    def get_path(self):
        pass