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
    return np.linalg.norm(poi1-poi2, ord=ord)

def compute_norm_path(path: list, ord: Union[float, int]) -> float:
    total = 0
    for i in range(1,len(path)):
        total += compute_norm(path[i-1], path[i], ord)
    return total

def compute_diversity(cfs: list) -> float:
    total = 0
    for i in range(len(cfs)):
        for j in range(i,len(cfs)):
            total += compute_norm(cfs[i], cfs[j], 2)
    return total