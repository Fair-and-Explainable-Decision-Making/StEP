from dice_ml import Data
from models.model_interface import ModelInterface
from data.data_interface import DataInterface
import pandas as pd
from recourse.recourse_interface import RecourseInterface
from recourse.step_lib import StEP, StEPRecourse
from recourse.dice_recourse import DiceRecourse
from recourse.face_recourse import FACERecourse
import numpy as np

def get_recourse_interface_by_name(recourse_name: str, model_interface: ModelInterface, data_interface: DataInterface, **kwargs) -> ModelInterface:
    if recourse_name == "StEP":
        return StEPRecourse(model_interface, data_interface, kwargs['k_directions'], kwargs['max_iterations'], confidence_threshold=kwargs['confidence_threshold'],
                            directions_rescaler=kwargs['directions_rescaler'], step_size=kwargs['step_size'],random_seed=kwargs['random seed'])
    elif recourse_name == "DiCE":
        return DiceRecourse(model_interface, data_interface, backend=kwargs["backend"], default_k=kwargs['k_directions'], 
                            confidence_threshold=kwargs['confidence_threshold'],random_seed=kwargs['random seed'])
    elif recourse_name == "FACE":
        return FACERecourse(model_interface, data_interface, k_directions=kwargs['k_directions'], distance_threshold=kwargs['direction_threshold'], 
                            confidence_threshold=kwargs['confidence_threshold'],weight_bias=kwargs['weight_bias'])
    else:
        raise Exception("Invalid recourse choice")

def constrain_one_hot_cat_featutes(poi: pd.DataFrame, data_interface: DataInterface):
    for feat_name, one_hot_feat_names in data_interface.get_encoded_categorical_feats().items():
        one_hot_feats_constrained = np.zeros_like(
            poi[one_hot_feat_names].values)
        one_hot_feats_constrained[np.arange(len(poi[one_hot_feat_names])),
                                    poi[one_hot_feat_names].values.argmax(1)] = 1
        poi[one_hot_feat_names] = one_hot_feats_constrained
    return poi