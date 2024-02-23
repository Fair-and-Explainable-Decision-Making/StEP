import pandas as pd
import numpy as np
import string
from typing import List
from sklearn.preprocessing import normalize
from pandas.core.common import flatten


def col_names_synthetic(s: str, num_feat: int) -> List[str]:
    """
    Simply creates a list of string + int n for column naming
    """
    return [s+str(i) for i in range(1, num_feat+1)]


def create_synthetic_data(n_samples: int, num_con_feat: int = 3, num_ord_cat_feat: int = 1,
                          ord_cat_num_unique: int = 6, num_binary_cat_feat: int = 1,
                          w: List[float] = [1.0, -1.0, 1.0, 1.0, -1.0]) -> pd.DataFrame:
    """
    Creates a sythetic dataset with specified number of samples, continuous features,
    ordinal features, and categorical features.

    Features are normalized and then passed to a sigmoid function with specified coefficients
    to produce a target label. 

    Parameters
    ----------
    n_samples : int
        Number of samples 
    num_con_feat : int
        Number of continous featues
    num_ord_cat_feat: int
        Number of ordinal (categorical) features
    ord_cat_num_unique : int
        Number of unique values for the ordinal features
    num_binary_cat_feat : int
        Number of binary categorical features
    w : List[float]
        List of weights for producing the target feature. Must be equal to the number of features.
    Returns
    -------
    XZOY_df : pd.DataFrame
        Dataframe of synthetic dataset

    """

    # Creates rows and features and combines them to one 2D numpy array
    feats = []
    if num_con_feat:
        X = np.random.random((n_samples, num_con_feat))
        feats.append(X)
    if num_binary_cat_feat:
        Z = (2*np.random.randint(2, size=(num_binary_cat_feat, n_samples))-1).T
        feats.append(Z)
    if num_ord_cat_feat:
        O = np.random.randint(ord_cat_num_unique, size=(
            num_ord_cat_feat, n_samples)).T
        feats.append(O)
    feats = np.hstack(feats)

    # normalization
    feats = normalize(feats, axis=0, norm='max')

    # create equivalent categorical columns but where the integers are instance letters
    d = dict(enumerate(string.ascii_uppercase))
    data_char_vals = []
    if num_con_feat:
        data_char_vals.append(X)
    if num_binary_cat_feat: 
        Z_char = np.vectorize(d.get)(((Z+1)/2).astype(int))
        data_char_vals.append(Z_char)
    if num_ord_cat_feat:
        O_char = np.vectorize(d.get)(O.astype(int))
        data_char_vals.append(O_char)

    print(feats)
    # calculate target features
    y = 1/(1 + np.exp(-feats @ w))
    Y = np.array(y >= 0.5, dtype=np.int32).reshape(-1, 1)
    print(Y)
    data_char_vals.append(Y)
    # takes the target feature and creates a new 2d numpy list with it, continuos features
    # and categorical features replaces with letters
    data_char_vals = np.hstack(data_char_vals)
    # create some column names and create a dataframe for our data
    cols = []
    if num_con_feat:
        X_cols = col_names_synthetic('Con', num_con_feat)
        cols.append(X_cols)
    if num_binary_cat_feat:
        Z_cols = col_names_synthetic('Cat', num_binary_cat_feat)
        cols.append(Z_cols)
    if num_ord_cat_feat:
        O_cols = col_names_synthetic('Ord', num_ord_cat_feat)
        cols.append(O_cols)
    
    cols.append("Target")
    cols = list(flatten(cols))
    print(cols)
    syn_data_df = pd.DataFrame(data_char_vals, columns=cols)
    syn_data_df = syn_data_df.astype({'Target': int})
    for i in range(1, num_con_feat+1):
        syn_data_df = syn_data_df.astype({'Con'+str(i): float})
    return syn_data_df
