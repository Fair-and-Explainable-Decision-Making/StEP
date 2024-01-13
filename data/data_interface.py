import pandas as pd
from typing import Sequence, Any, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np

#TODO: if binary labels set to 0 or 1, also allow for flipping if necessary between pos and neg labels to match our convention
class DataInterface():
    def __init__(self, data: pd.DataFrame, file_path: str, continuous_features: Sequence[str],
                ordinal_features: Sequence[str], categorical_features: Sequence[str], immutable_features: Sequence[str],
                target_feature: str, target_mapping: Any = None, scaling_method: str = "MinMax", encoding_method: str = "OneHot",
                pos_label: int = 1, file_header_row: int = 0, dropped_columns: list = [],
                unidirection_features: (Sequence[str],Sequence[str]) = ([],[]),
                feature_orders: dict = {}
                ):
        """
        Creates a data interface with the specified, continuous features, ordinal features, and categorical features.

        Features can be normalized, standardized, or encoded as desired. Additionally data can be split into train and test data.

        Parameters
        ----------
        data : pd.DataFrame
            A dataframe of the data to load. If None, we will attempt to load using file_path
        file_path : str
            The path to load your data from. Not required with data is not None.
        continuous_features : Sequence[str]
            A sequence of the continous features.
        ordinal_features : Sequence[str]
            A sequence of the ordinal features.
        categorical_features : Sequence[str]
            A sequence of the categorical/nominal features.
        immutable_features : Sequence[str]
            A sequence of the immutable features.
        target_feature : str
            The column name of the target feature
        target_mapping : Any
            A mapping of the target features to some values
        scaling_method : str
            String indicating how you with to scale your data. 
            Options are MaxMin normalization and standardization. 
        encoding_method : str
            Encoding method for categorical features. Only one-hot for now.
        file_header_row : int
            Row the headers are on when a file and its path is given for reading.
        dropped_columns : list
            List of columns to drop from the dataset.
        unidirection_features: (Sequence[str],Sequence[str])
            Tuple of two lists. 1st list are features that only decrease 
            and 2nd list are for those that only increase.
        feature_orders: dict
            Dictionary of keys representing feature names and 
            (dict)values as a list representing the order of the feature's values.
        """
        self._continuous_features = continuous_features
        self._ordinal_features =  ordinal_features
        self._categorical_features = categorical_features
        self._immutable_features = immutable_features
        self._target_feature = target_feature
        self._target_mapping = target_mapping
        self._scaling_method = scaling_method
        self._encoding_method = encoding_method
        self._file_path = file_path
        
        if (file_path is None or file_path == "") and isinstance(data, pd.DataFrame):
            df = data
        elif file_path is not None and file_path != "" and file_path.endswith(".csv"):
            df = pd.read_csv(file_path, header=file_header_row)
        elif file_path is not None and file_path != "" and file_path.endswith(".xls"):

            df = pd.read_excel(file_path, header=file_header_row)
        else:
            raise Exception("No data and valid file path provided")
        self.dataset = df.copy()
        self._pos_label = pos_label
        self._unidirection_features = unidirection_features

        """
        Necessary preprocessing for dropping undesired columns and mapping 
        labels to 0 (negative) and 1 (positive).
        """
        df = df.drop(columns=dropped_columns)
        if target_mapping is None:
            neg_labels = np.delete(pd.unique(df[target_feature]), 
                                   np.where(pd.unique(df[target_feature]) == pos_label))
            d1 = dict.fromkeys(neg_labels, 0)
            d2 = {pos_label: 1}
            target_mapping = {**d1, **d2}
        df[target_feature] = df[target_feature].map(target_mapping)
        for feat_name, value_order in feature_orders.items():
            feat_mapping = {k: v for v, k in enumerate(value_order)}
            df[feat_name] = df[feat_name].map(feat_mapping)
        self._labels_df = df[self._target_feature]
        self._features_df = df[df.columns[df.columns != target_feature]]
        
    def split_data(self, validation_size:float = 0.15, test_size: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame ,pd.DataFrame]:
        """
        Splits data into X and y and train and test data. Should be used as the last step when setting up data.

        Parameters
        ----------
        validation_size : float
            Between 0.0 and 1.0 and represents the proportion of the dataset to include in the validation split.
        test_size : float
            Between 0.0 and 1.0 and represents the proportion of the dataset to include in the test split.
        Returns
        -------
        features_train, features_valid, features_test, labels_train, labels_valid, labels_test: tuple
            Dataframes of non-target and target features seperated also split into training, validation, and testing data.
        """
        self._features_train, self._features_test, self._labels_train, self._labels_test = train_test_split(
            self._features_df, self._labels_df, test_size=validation_size+test_size, random_state=42)
        test_ratio = test_size/(validation_size+test_size)
        self._features_valid, self._features_test, self._labels_valid, self._labels_test = train_test_split(
            self._features_test, self._labels_test, test_size=test_ratio, random_state=42)
        return self._features_train, self._features_valid, self._features_test, self._labels_train,self._labels_valid, self._labels_test
    
    def scale_data(self)-> pd.DataFrame:
        """
        Scales the data according to what was specified in the constructor. If MinMax and Standard scaler were
        not specifies, no scaling will happen.

        For now we will not scale the target feature.

        Returns
        -------
        self._features_df : pd.DataFrame
            Dataframe of scaled data.
        """
        if self._scaling_method == "MinMax":
            self._scaler = MinMaxScaler()
        elif self._scaling_method == "Standard":
            self._scaler = StandardScaler()
        else:
            return self._features_df
        #in classification we don't want to scale the target column
        scaled = self._scaler.fit_transform(self._features_df) 
        self._features_df = pd.DataFrame(scaled, columns=self._features_df.columns,index=self._features_df.index)
        return self._features_df

    def encode_data(self) -> pd.DataFrame:
        """
        Applies one hot encoding to categorical features.

        Returns
        -------
        self._features_df : pd.DataFrame
            Dataframe of features data after one-hot encoding.
        """ 
        if self._encoding_method == "OneHot":
            self._features_df = pd.get_dummies(self._features_df, columns = self._categorical_features) 
        return self._features_df

    def get_data(self):
        """
        Simple way to return the processed data with labels column.
        
        Returns
        -------
        pd.concat[self._features_df, self._labels_df] : pd.DataFrame
            Dataframe of potentially scaled and encoded data with features and labels.
        """
        return pd.concat[self._features_df, self._labels_df]
    
    def get_train_test_split(self):
        """
        Returns the split data if it has been created.
        """
        if (self._features_train is not None or self._features_test is not None 
            or self._labels_train is not None or self._labels_test is not None):
            return self._features_train, self._features_test, self._labels_train, self._labels_test
        else:
            raise Exception("One of your splits are None")

    def get_scaler(self):
        """
        Returns the scaler if it has been created and fit for the data.
        """
        if self._scaler is not None:
            return self._scaler
        else:
            raise Exception("You have not scaled your data.")
    @property
    def categorical_features(self) -> Sequence[str]:
        return self._categorical_features
    
    @property
    def ordinal_features(self) -> Sequence[str]:
        return self._ordinal_features

    @property
    def continuous_features(self) -> Sequence[str]:
        return self._continuous_features

    @property
    def immutable_features(self) -> Sequence[str]:
        return self._immutable_features

    @property
    def target_feature(self) -> str:
        return self._target_feature

    @property
    def pos_label(self) -> int:
        return self._pos_label
    
    @property
    def unidirection_features(self) -> (Sequence[str], Sequence[str]):
        return self._unidirection_features
    
    def copy_change_data(self, data: pd.DataFrame):
        return DataInterface(data, self._file_path, self._continuous_features, self._ordinal_features,
                             self._categorical_features, self._immutable_features, self._target_feature,
                             self._target_mapping, self._scaling_method, self._encoding_method,
                             self._pos_label)
