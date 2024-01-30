from models.model_interface import ModelInterface
from data.data_interface import DataInterface
import pandas as pd
from recourse_interface import RecourseInterface
from recourse.step_lib import StEP, StEPRecourse
from recourse.dice_recourse import DiceRecourse

def get_recourse_interface_by_name(recourse_name: str, model_interface: ModelInterface, data_interface: DataInterface, **kwargs) -> ModelInterface:
    if recourse_name == "StEP":
        return StEPRecourse(model_interface, data_interface, kwargs['num_clusters'], kwargs['max_iterations'], confidence_threshold=kwargs['confidence_threshold'],
                            directions_rescaler=kwargs['directions_rescaler'], step_size=kwargs['step_size'])
    elif recourse_name == "DiCE":
        return DiceRecourse(model_interface, data_interface, backend=kwargs["backend"])
    elif recourse_name == "FACE":
        pass
    else:
        raise Exception("Invalid recourse choice")
    