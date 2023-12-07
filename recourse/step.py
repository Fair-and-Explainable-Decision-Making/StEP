from ..models.model_interface import ModelInterface
from ..data.data_interface import DataInterface
import pandas as pd
from recourse_interface import RecourseInterface
from step_lib import StEP
from typing import Optional

#binary should be full pipeline data->model->recourse->eval

class StEPRecourse(RecourseInterface):
    def __init__(self, model: ModelInterface, data_interface:DataInterface,
        num_clusters: int, use_train_data: bool = True,
        confidence_threshold: Optional[float] = None, random_seed: Optional[int] = None
    ) -> None:
        
        self.StEP_instance = StEP(num_clusters, data_interface, model, use_train_data,
                                  confidence_threshold, random_seed)
    
    def get_counterfactuals(self, poi: pd.DataFrame) -> pd.Dataframe:
        cfs = []
        paths = self.StEP_instance.compute_paths(poi)
        for p in paths:
            cfs.append(p[-1])
        return pd.concat(cfs, ignore_index=True)

    def get_paths(self, poi: pd.DataFrame):
        return self.StEP_instance.compute_paths(poi)