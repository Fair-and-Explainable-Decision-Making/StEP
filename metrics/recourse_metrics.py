import pandas as pd
import numpy as np
from typing import Union


def compute_norm(poi1: pd.DataFrame, poi2: pd.DataFrame,
                 ord: Union[float, int], sparsity_epsilon=1e-5) -> float:
    """
    Ord=0 -> 0-"Norm" for checking num of features changes
    Ord=1 -> 1-Norm for Manhattan distance
    Ord=2 -> 2-Norm for Euclidean distance
    """
    if ord == 0:
        return (np.abs(poi1 - poi2) > sparsity_epsilon).values.sum()
    return np.linalg.norm(poi1.values-poi2.values, ord=ord)


def compute_norm_path(path: list, ord: Union[float, int]) -> float:
    total = 0
    for i in range(1, len(path)):
        total += compute_norm(path[i-1], path[i], ord)
    return total


def compute_diversity(poi: pd.DataFrame,cfs: list) -> float:
    total = 0
    if len(cfs)==0:
        return np.nan
    max_norm = 0
    nan_ct = 0
    for i in range(len(cfs)):
        if cfs[i] is None or cfs[i] is np.nan:
            nan_ct +=1
            if nan_ct >= len(cfs) - 1:
                return np.nan
        else:
            if max_norm < compute_norm(poi, cfs[i], ord=2):
                max_norm = compute_norm(poi, cfs[i], ord=2)
            for j in range(i, len(cfs)):
                if not (cfs[j] is None or cfs[j] is np.nan):
                    total += compute_norm(cfs[i], cfs[j], 2)
    return total/max_norm
