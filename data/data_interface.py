import pandas as pd
from typing import Sequence, Any, Tuple
import os
import pathlib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from synthetic_data import create_synthetic_data

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
        X_features = df.columns[df.columns != target_feature]
        self._df_y = df[self._target_feature]
        self._df_X = df[X_features]
        self._df_y_init = self._df_y.copy()
        self._df_X_init = self._df_X.copy()
        self._X_train = None
        self._X_test = None
        self._y_train = None
        self._y_test = None
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
        self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(self._df_X, self._df_y, test_size=test_size, random_state=42)
        return self._X_train, self._X_test, self._y_train, self._y_test
    
    def scale_data(self)-> pd.DataFrame:
        """
        Scales the data according to what was specified in the constructor. If MinMax and Standard scaler were
        not specifies, no scaling will happen.

        For now we will not scale the target feature.

        Returns
        -------
        self._df_X : pd.DataFrame
            Dataframe of scaled data.
        """
        if self._scaling_method == "MinMax":
            scaler = MinMaxScaler()
        elif self._scaling_method == "Standard":
            scaler = StandardScaler()
        else:
            return self._df_X
        #in classification we don't want to scale the target column
        scaled = scaler.fit_transform(self._df_X) 
        self._df_X = pd.DataFrame(scaled, columns=self._df_X.columns,index=self._df_X.index)
        return self._df_X

    def encode_data(self) -> pd.DataFrame:
        """
        Applies one hot encoding to categorical features
        """ 
        if self._encoding_method == "OneHot":
            self._df_X = pd.get_dummies(self._df_X, columns = self._categorical_features) 
        return self._df_X

    def save_data(self, dataset_filepath: str) -> None:
        """Saves the data to local disk as a csv.

        Args:
            dataset_filepath: The filepath to save the data to.
        """
        if not os.path.exists(pathlib.Path(dataset_filepath).parent):
            os.makedirs(pathlib.Path(dataset_filepath).parent)
        df = pd.concat[self._df_X, self._df_y]
        df.to_csv(dataset_filepath, header=True, index=False)

    def get_data(self):
        """
        Simple way to return the data.
        """
        return pd.concat[self._df_X, self._df_y]
    
    def get_init_data(self):
        """
        Simple way to return the data from before any changes to it.
        """
        return pd.concat[self._df_X_init, self._df_y_init]
    
    def get_split_data(self):
        return self._X_train, self._X_test, self._y_train, self._y_test

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
    
if __name__ == "__main__":
    df = create_synthetic_data(1000)
    cols = list(df.columns)
    targ = cols[-1]
    cont = cols[:3]
    ord = [cols[4]]
    cat = cols[3:5]
    imm =  [cols[3]]
    
    
    di = DataInterface(df, None, cont, ord, cat, imm, targ)
    di.encode_data()
    di.scale_data()
    X_train, X_test, y_train, y_test = di.split_data()
    print(X_train.head(5))
    print(y_train.head(5))
    print(X_train.shape)
    print(X_test.shape)
    print(X_train.index)
    print(y_train[y_train == 1].index)
    print(X_train.loc[y_train[y_train == 1].index])
