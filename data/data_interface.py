import pandas as pd
from typing import Sequence, Any, Tuple, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import itertools
import re
from sklearn.preprocessing import OneHotEncoder
from typing import Union

# TODO: if binary labels set to 0 or 1, also allow for flipping if necessary between pos and neg labels to match our convention


class DataInterface():
    def __init__(self, data: pd.DataFrame, file_path: str, continuous_features: Sequence[str],
                 ordinal_features: Sequence[str], categorical_features: Sequence[str], immutable_features: Sequence[str],
                 target_feature: str, target_mapping: Any = None, encoding_method: str = "OneHot",
                 pos_label: int = 1, file_header_row: int = 0, dropped_columns: list = [],
                 unidirection_features: List[Sequence[str]] = ([], []),
                 ordinal_features_order: dict = {}, data_name: str = "noname", prot_attrs = []
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
        ordinal_features_order: dict
            Dictionary of keys representing ordinal feature names and 
            (dict)values as a list representing the order of the feature's values.
        """
        self._continuous_features = continuous_features
        self._ordinal_features = ordinal_features
        self._categorical_features = categorical_features
        self._immutable_features = immutable_features
        self._target_feature = target_feature
        self._target_mapping = target_mapping
        self._encoding_method = encoding_method
        self._file_path = file_path
        self.name = data_name
        self._prot_attrs = prot_attrs 

        if (file_path is None or file_path == "") and isinstance(data, pd.DataFrame):
            df = data
        elif file_path is not None and file_path != "" and file_path.endswith(".csv"):
            df = pd.read_csv(file_path, header=file_header_row)
        elif file_path is not None and file_path != "" and file_path.endswith(".xls"):

            df = pd.read_excel(file_path, header=file_header_row)
        else:
            raise Exception("No data and valid file path provided")
        df = df.drop(columns=dropped_columns)
        self.dataset = df.copy()
        self._pos_label = pos_label
        self._unidirection_features = unidirection_features
        if (list(set(unidirection_features[0]) & set(categorical_features))
                or list(set(unidirection_features[1]) & set(categorical_features))):
            raise Exception("""Categorical features cannot be unidirectional.
                            Did you mean to make it ordinal?""")

        """
        Necessary preprocessing for dropping undesired columns and mapping 
        labels to 0 (negative) and 1 (positive).
        """
        
        if target_mapping is None:
            neg_labels = np.delete(pd.unique(df[target_feature]),
                                   np.where(pd.unique(df[target_feature]) == pos_label))
            d1 = dict.fromkeys(neg_labels, 0)
            d2 = {pos_label: 1}
            target_mapping = {**d1, **d2}
        df[target_feature] = df[target_feature].map(target_mapping)
        if ordinal_features: 
            ordinal_features_list = ordinal_features.copy()
            for feat_name, value_order in ordinal_features_order.items():
                ordinal_features_list.remove(feat_name)
                feat_mapping = {k: v for v, k in enumerate(value_order)}
                df[feat_name] = df[feat_name].map(feat_mapping)
            if ordinal_features_list:
                raise Exception("The following ordinal features", ordinal_features_list,
                                "did not have an ordering given.")
        self._labels_df = df[self._target_feature]
        self._features_df = df[df.columns[df.columns != target_feature]]
        self._feature_columns = df.columns[df.columns != target_feature]

    def split_data(self, validation_size: float = 0.15, test_size: float = 0.15,random_state=None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
            self._features_df, self._labels_df, test_size=validation_size+test_size, random_state=random_state)
        test_ratio = test_size/(validation_size+test_size)
        self._features_valid, self._features_test, self._labels_valid, self._labels_test = train_test_split(
            self._features_test, self._labels_test, test_size=test_ratio,random_state=random_state)
        return self._features_train, self._features_valid, self._features_test, self._labels_train, self._labels_valid, self._labels_test

    def scale_data(self, scaling_method: str) -> pd.DataFrame:
        """
        Scales the data according to what was specified in the constructor. If MinMax and Standard scaler were
        not specifies, no scaling will happen.

        For now we will not scale the target feature.

        Parameters
        ----------
        scaling_method : str
            String indicating how you with to scale your data. 
            Options are MaxMin normalization and standardization. 
        Returns
        -------
        self._features_df : pd.DataFrame
            Dataframe of scaled data.
        """
        self._scaling_method = scaling_method
        if self._scaling_method == "MinMax":
            self._scaler = MinMaxScaler()
        elif self._scaling_method == "Standard":
            self._scaler = StandardScaler()
        else:
            raise Exception(
                "Invalid scaler name. Pick one of MinMax or Standard.")
        # in classification we don't want to scale the target column
        if self._encoded_categorical_feats_dict is not None:
            feats_to_drop = sum(list(self._encoded_categorical_feats_dict.values()),[])
        else: 
            feats_to_drop = self._categorical_features
        non_cat_features_df = self._features_df.copy().drop(columns=feats_to_drop)
        self._scaled_features = non_cat_features_df.columns.values
        scaled = self._scaler.fit_transform(non_cat_features_df)
        self._features_df[non_cat_features_df.columns] = scaled
        return self._features_df

    def encode_data(self) -> pd.DataFrame:
        """
        Applies one hot encoding to categorical features.

        Returns
        -------
        self._features_df : pd.DataFrame
            Dataframe of features data after one-hot encoding.
        """
        binary_cols_to_drop=[]
        for col in self._categorical_features:
            if len(self._features_df[col].unique())==2:
                binary_cols_to_drop.append(str(col)+"_"+str(np.sort(self._features_df[col].unique())[0]))
        if self._encoding_method == "OneHot":
            self._features_df = pd.get_dummies(
                self._features_df, columns=self._categorical_features)
            self._features_df = self._features_df.drop(columns=binary_cols_to_drop)
        encoded_cat_feats = self._features_df.columns.intersection(
            self._features_df.columns.symmetric_difference(self._feature_columns))
        
        self._encoded_categorical_features = encoded_cat_feats.copy()
        self._encoded_categorical_feats_dict = {k: list(v)for k, v in itertools.groupby(
            encoded_cat_feats, key=lambda x: re.match('(.*)(_[^_]+)', x).group(1))}
        
        self._encoded_immutable_features = self._immutable_features.copy()
        self._prot_attr_cat_features = []
        for feat_name, one_hot_feat_names in self.get_encoded_categorical_feats().items():
                if feat_name in self._immutable_features and feat_name in self._prot_attrs:
                    self._encoded_immutable_features.remove(feat_name)
                    for name in one_hot_feat_names:
                        self._encoded_immutable_features.append(name)
                        self._prot_attr_cat_features.append(name)
                elif feat_name in self._immutable_features:
                    self._encoded_immutable_features.remove(feat_name)
                    for name in one_hot_feat_names:
                        self._encoded_immutable_features.append(name)
        return self._features_df

    def get_data(self):
        """
        Simple way to return the processed data with labels column.

        Returns
        -------
        pd.concat[self._features_df, self._labels_df] : pd.DataFrame
            Dataframe of potentially scaled and encoded data with features and labels.
        """
        return pd.concat([self._features_df, self._labels_df], axis=1)

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
            return None

    def get_encoded_categorical_feats(self, with_original_names: bool = True) -> Union[dict,list]:
        """
        Returns a dict of original column name to list of encoded column names mappings.
        """
        if self._encoded_categorical_feats_dict is not None:
            if with_original_names:
                return self._encoded_categorical_feats_dict
            else:
                return self._encoded_categorical_features
        else:
            return None
    
    def get_processed_prot_feats(self) -> list:
        if self._prot_attr_cat_features is not None:
            return self._prot_attr_cat_features
        else:
            return self._prot_attrs
    
    def get_processed_immutable_feats(self) -> list:
        if self._encoded_immutable_features is not None:
            return self._encoded_immutable_features
        else:
            return self._immutable_features

    def get_scaled_features(self) -> list:
        if self._scaled_features is not None:
            return self._scaled_features
        else:
            return None
        
    def get_processed_features(self) -> list:
        return self._features_df.columns.values
    
    def inverse_scale_ordinal(self, feat_name: str, data_point: pd.DataFrame):
        t = np.sort(self.get_data()[feat_name].unique())-self.get_data()[feat_name].unique().min()
        base = t[1]-t[0]
        x=data_point[feat_name]
        return (base * round(x/base))

    @property
    def features(self) -> Sequence[str]:
        return self._feature_columns

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
    def unidirection_features(self) -> List[Sequence[str]]:
        return self._unidirection_features

    def copy_change_data(self, data: pd.DataFrame):
        return DataInterface(data, self._file_path, self._continuous_features, self._ordinal_features,
                             self._categorical_features, self._immutable_features, self._target_feature,
                             self._target_mapping, self._scaling_method, self._encoding_method,
                             self._pos_label)
