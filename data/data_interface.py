import pandas as pd
from typing import Sequence, Any, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

#TODO: if binary labels set to 0 or 1, also allow for flipping if necessary between pos and neg labels to match our convention
class DataInterface():
    def __init__(self, data: pd.DataFrame, file_path: str, continuous_features: Sequence[str],
                ordinal_features: Sequence[str], categorical_features: Sequence[str], immutable_features: Sequence[str],
                target_feature: str, target_mapping: Any = None, scaling_method: str = "MinMax", encoding_method: str = "OneHot",
                pos_label: int = 1):
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
            String indicating how you with to scale your data. Options are MaxMin normalization and standardization 
        encoding_method : str
            Encoding method for categorical features. Only one-hot for now.
        """
        self._continuous_features = continuous_features
        self._ordinal_features =  ordinal_features
        self._categorical_features = categorical_features
        self._immutable_features = immutable_features
        self._target_feature = target_feature
        self._target_mapping = target_mapping
        self._scaling_method = scaling_method
        self._encoding_method = encoding_method
        if (file_path is None or file_path == "") and isinstance(data, pd.DataFrame):
            df = data
        elif file_path is not None and file_path != "":
            df = pd.read_csv(file_path)
        else:
            raise Exception("No data and file path provided")
        self.dataset = df.copy()
        self._labels_df = df[self._target_feature]
        self._features_df = df[df.columns[df.columns != target_feature]]
        self._pos_label = pos_label
    
    def split_data(self, test_size: float = 0.3) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame ,pd.DataFrame]:
        """
        Splits data into X and y and train and test data. Should be used as the last step when setting up data.

        Parameters
        ----------
        test_size : float
            Between 0.0 and 1.0 and represents the proportion of the dataset to include in the test split.
        Returns
        -------
        X_train, X_test, y_train, y_test : Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame ,pd.DataFrame]
            Dataframes of non-target and target features seperated also split into training and testing data.
        """
        #TODO: add validation split
        self._features_train, self._features_test, self._labels_train, self._labels_test = train_test_split(self._features_df, self._labels_df, test_size=test_size, random_state=42)
        return self._features_train, self._features_test, self._labels_train, self._labels_test
    
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
            scaler = MinMaxScaler()
        elif self._scaling_method == "Standard":
            scaler = StandardScaler()
        else:
            return self._features_df
        #in classification we don't want to scale the target column
        scaled = scaler.fit_transform(self._features_df) 
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
    
    def get_split_data(self):
        """
        Returns the split data if it has been created.
        """
        if (self._features_train is not None or self._features_test is not None 
            or self._labels_train is not None or self._labels_test is not None):
            return self._features_train, self._features_test, self._labels_train, self._labels_test
        else:
            raise Exception("One of your splits are None")

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
    
