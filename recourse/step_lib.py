from __future__ import annotations
from typing import Optional
import numpy as np
from models import model_interface
from data import data_interface
import pandas as pd
from sklearn.cluster import KMeans
from recourse_interface import RecourseInterface


class StEP:
    def __init__(
        self, k_directions: int, data_inter: data_interface.DataInterface,
        model: model_interface.ModelInterface, use_train_data: bool = True, 
        confidence_threshold: Optional[float] = None, random_seed: Optional[int] = None,
        step_size: Optional[float] = None, max_iterations: Optional[int] = None
    ):
        
        self.k_directions = k_directions
        self._model = model
        if use_train_data:
            X_data, y_data, _, _ = data_inter.get_split_data()
        else:
            _, _, X_data, y_data = data_inter.get_split_data()
        
        #comment out data_inter 
        self._data_inter = data_inter
        self.data = self._process_data(X_data, y_data, confidence_threshold=confidence_threshold)
        self.clusters_assignments, self.cluster_centers = self._cluster_data(
                self.data, self.k_directions, random_seed=random_seed
            )
        self.step_size = step_size
        self.max_iterations = max_iterations

    #features, labels
    def _process_data(self, X_data: pd.DataFrame, y_data: pd.Series, 
        confidence_threshold: Optional[float] = None) -> pd.DataFrame:
        
        positive_data = X_data.loc[y_data[y_data == 1].index]
        if confidence_threshold:
            probs = self._model.predict_proba(positive_data.values)
            positive_confident_df = pd.Series(probs.flatten(), index=positive_data.index)
            positive_confident_df = positive_confident_df[positive_confident_df >= confidence_threshold]
        if len(positive_confident_df) == 0:
            raise ValueError(
                "Dataset is empty after excluding negative outcome examples."
            )
        return positive_confident_df

    def _cluster_data(self, data: pd.DataFrame, k_directions: int,
        random_seed: Optional[int] = None) -> (pd.DataFrame, np.ndarray):
        
        km = KMeans(n_clusters=k_directions, random_state=random_seed)
        cluster_assignments = np.array(km.fit_predict(data)).reshape(-1,1)
        cluster_assignments_df = pd.DataFrame(
            index=data.index,
            columns=["datapoint_cluster"],
            data=cluster_assignments
        )
        cluster_centers = km.cluster_centers_
        return cluster_assignments_df, cluster_centers
    
    def compute_all_directions(self, poi: pd.DataFrame) -> list:
        directions = []
        for cluster_index in range(self.k_directions):
            cluster_data = self.data.loc[self.clusters_assignments[self.clusters_assignments == cluster_index].index]
            directions.append(self.compute_unnormalized_direction(poi, cluster_data))
        return directions
    
    def compute_k_direction(self, poi: pd.DataFrame, k: int) -> pd.DataFrame:
        cluster_data = self.data.loc[self.clusters_assignments[self.clusters_assignments == k].index]
        return self.compute_unnormalized_direction(poi, cluster_data)
        
    
    def compute_unnormalized_direction(self, poi: pd.DataFrame, cluster_data: pd.DataFrame) -> pd.DataFrame:
        diff = cluster_data.values - poi.values
        dist = np.sqrt(np.power(diff, 2).sum(axis=1))
        alpha_val = self.volcano_alpha(dist)
        if np.isnan(alpha_val).any():
            raise RuntimeError(f"Alpha function returned NaN values: {alpha_val}")
        direction = diff.T @ alpha_val
        return pd.DataFrame(direction, columns=cluster_data.columns)
    
    def compute_direction(self, poi: pd.DataFrame, cluster_data: pd.DataFrame) -> pd.DataFrame:
        direction = self.compute_unnormalized_direction(poi, cluster_data)
        if self.step_size:
            direction = self.constant_step_size(direction, self.step_size)
        return direction
    
    def compute_paths_branching(self, poi: pd.DataFrame, max_interations: int):
        """if max_interations == 0 or self._model.predict(poi.values)[0] == 1:
            return 
        directions = self.compute_all_directions(poi)
        for d in directions:
            poi = poi.add(d, fill_value=0)
            self.compute_paths_branching(poi, max_interations-1)"""
        pass

    def compute_paths(self, poi: pd.DataFrame):
        paths = []
        directions = self.compute_all_directions(poi)
        for k, d in enumerate(directions):
            new_poi = poi.copy()
            path = [new_poi]
            drct = d.copy()
            for i in self.max_iterations:
                new_poi = new_poi.add(drct, fill_value=0)
                path.append(new_poi)
                drct = self.compute_k_direction(new_poi, k)
            paths.append(path)
        return paths

    
    def volcano_alpha(self, dist: np.ndarray, cutoff=0.5, degree=2) -> np.ndarray:
        return 1 / np.where(dist <= cutoff, cutoff, dist) ** degree
    
    def constant_step_size(self,
        direction: pd.DataFrame, step_size: float = 1
    ) -> pd.DataFrame:
        
        """Rescales a vector to a given fixed size measured by L2 norm.

        Args:
            direction: The vector to rescale.
            step_size: The target L2 norm of the rescaled vector.

        Returns:
            A new vector with direction equal to the original but rescaled to the
            magnitude given by `step_size`.
        """
        _MIN_DIRECTION = 1e-32
        """Numbers smaller than this won't be used as the denominator during
        division."""
        normalization = np.linalg.norm(direction)
        if normalization == 0:
            return direction
        if normalization <= _MIN_DIRECTION:
            normalization = _MIN_DIRECTION
        return (step_size * direction) / normalization

class StEPRecourse(RecourseInterface):
    def __init__(self, model: ModelInterface, data_interface:DataInterface,
        num_clusters: int, use_train_data: bool = True,
        confidence_threshold: Optional[float] = None, random_seed: Optional[int] = None
    ) -> None:
        
        self.StEP_instance = StEP(num_clusters, data_interface, model, use_train_data,
                                  confidence_threshold, random_seed)
    
    def get_counterfactuals(self, poi: pd.DataFrame) -> pd.Dataframe:
        cfs = []
        paths = self.StEP_instance.compute_paths(poi)
        for p in paths:
            cfs.append(p[-1])
        return pd.concat(cfs, ignore_index=True)

    def get_paths(self, poi: pd.DataFrame):
        return self.StEP_instance.compute_paths(poi)
    
if __name__ == "__main__":
    from ..models.model_interface import ModelInterface
    from ..data.data_interface import DataInterface
    from ..data.synthetic_data import create_synthetic_data

    df = create_synthetic_data(1000)
    cols = list(df.columns)
    targ = cols[-1]
    cont = cols[:3]
    ord = [cols[4]]
    cat = cols[3:5]
    imm =  [cols[3]]
    di = DataInterface(df, None, cont, ord, cat, imm, targ)