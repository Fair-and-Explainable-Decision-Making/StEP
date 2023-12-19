import pandas as pd
import numpy as np
import string
from typing import List
from sklearn.preprocessing import normalize

def col_names_synthetic(s: str, num_feat: int) -> List[str]:
    """
    Simply creates a list of string + int n for column naming
    """
    return [s+str(i) for i in range(1,num_feat+1)]

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
    X = np.random.random((n_samples, num_con_feat))
    Z = np.random.randint(2, size=(num_binary_cat_feat,n_samples)).T
    O = np.random.randint(ord_cat_num_unique, size=(num_ord_cat_feat,n_samples)).T
    XZO = np.hstack([X,Z,O])

    #normalization
    XZO = normalize(XZO, axis=0, norm='max')

    #create equivalent categorical columns but where the integers are instance letters
    d = dict(enumerate(string.ascii_uppercase))
    Z_char = np.vectorize(d.get)(Z.astype(int))
    O_char = np.vectorize(d.get)(O.astype(int))

    #calculate target features
    y = 1/(1 + np.exp(-XZO @ w))
    Y = np.array(y >= 0.5, dtype=np.int32).reshape(-1,1)   
    #takes the target feature and creates a new 2d numpy list with it, continuos features
    #and categorical features replaces with letters
    XZOY_char = np.hstack([X,Z_char,O_char,Y])
    #create some column names and create a dataframe for our data
    X_cols = col_names_synthetic('X', num_con_feat)
    Z_cols = col_names_synthetic('Z', num_binary_cat_feat)
    O_cols = col_names_synthetic('O', num_ord_cat_feat)
    XZOY_df = pd.DataFrame(XZOY_char, columns=X_cols+Z_cols+O_cols+['Y'])
    XZOY_df = XZOY_df.astype({'Y': int})
    for i in range(1,num_con_feat+1):
        XZOY_df = XZOY_df.astype({'X'+str(i): float})
    return XZOY_df
