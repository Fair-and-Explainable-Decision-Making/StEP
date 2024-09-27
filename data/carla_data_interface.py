from carla import Data
from data_interface import DataInterface
import pandas as pd

# Custom data set implementations need to inherit from the Data interface


class CarlaDataInterface(Data):
    def __init__(self, data_interface: DataInterface):
        # The data set can e.g. be loaded in the constructor
        self._data_interface = data_interface

    # List of all categorical features
    @property
    def categorical(self):
        return self._data_interface.categorical_features

    # List of all continuous features
    @property
    def continuous(self):
        return self._data_interface.continuous_features

    # List of all immutable features which
    # should not be changed by the recourse method
    @property
    def immutables(self):
        return self._data_interface.immutable_features

    # Feature name of the target column
    @property
    def target(self):
        return self._data_interface.target_feature

    # The full dataset
    @property
    def df(self):
        return pd.concat([self.df_train, self.df_test], axis=0)

    # The training split of the dataset
    @property
    def df_train(self):
        features_train, _, labels_train, _ = self._data_interface.get_train_test_split()
        return pd.concat([features_train, labels_train], axis=1)

    # The test split of the dataset
    @property
    def df_test(self):
        _, features_test, _, labels_test = self._data_interface.get_train_test_split()
        return pd.concat([features_test, labels_test], axis=1)

    # Data transformation, for example normalization of continuous features
    # and encoding of categorical features
    def transform(self, df):
        return None

    # Inverts transform operation
    def inverse_transform(self, df):
        return None
