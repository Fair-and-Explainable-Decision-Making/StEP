from carla import MLModel
import pandas as pd
from typing import Union
import sklearn
import numpy as np
import torch
from data.data_interface import DataInterface
from model_interface import ModelInterface

from models.pytorch_wrapper import PyTorchModel
# Custom black-box models need to inherit from
# the MLModel interface


class CarlaModelInterface(MLModel):
    def __init__(self, model_interface: ModelInterface, data_carla, data_interface: DataInterface):
        super().__init__(data_carla)
        # The constructor can be used to load or build an
        # arbitrary black-box-model
        self._model_interface = model_interface
        self._data_carla = data_carla
        self._data_interface = data_interface

    # List of the feature order the ml model was trained on
    @property
    def feature_input_order(self):
        return self._data_interface.features.values

    # The ML framework the model was trained on
    @property
    def backend(self):
        return "pytorch"

    # The black-box model object
    @property
    def raw_model(self):
        return self._model_interface.get_model()

    # The predict function outputs
    # the continuous prediction of the model
    def predict(self, x):
        return self._model_interface.predict(x)

    # The predict_proba method outputs
    # the prediction as class probabilities
    def predict_proba(self, x):
        return self._model_interface.predict_proba(x)
