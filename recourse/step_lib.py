from __future__ import annotations
from typing import Optional, Sequence
import numpy as np
from models import model_interface
from data import data_interface
import pandas as pd
from sklearn.cluster import KMeans
from recourse.recourse_interface import RecourseInterface


class StEP:
    """
    TODO: docstring
    """

    def __init__(
        self, k_directions: int, data_inter: data_interface.DataInterface,
        model: model_interface.ModelInterface, max_iterations: int,
        use_train_data: bool = True, confidence_threshold: Optional[float] = None,
        random_seed: Optional[int] = None, step_size: Optional[float] = None,
        directions_rescaler: str = "normalize", special_cluster_data = None
    ):

        if directions_rescaler == "constant step size" and step_size == None:
            raise ("Step size required for constant step size rescaler")
        if directions_rescaler not in [None, "None", "normalize", "constant step size"]:
            raise ("Invalid direction rescaler")
        self.k_directions = k_directions
        self._model = model
        if special_cluster_data:
            features = special_cluster_data[0]
            labels = special_cluster_data[1]
        elif use_train_data:
            features, _, labels, _ = data_inter.get_train_test_split()
        else:
            _, features, _, labels = data_inter.get_train_test_split()
        self.processed_data = self._process_data(
            features, labels, confidence_threshold=confidence_threshold)
        self.clusters_assignments, self.cluster_centers = self._cluster_data(
            self.processed_data, self.k_directions, random_seed=random_seed
        )
        self.step_size = step_size
        self.max_iterations = max_iterations
        self.directions_rescaler = directions_rescaler
        self._data_interface = data_inter
        self._confidence_threshold = confidence_threshold

    def _process_data(self, features: pd.DataFrame, labels: pd.Series,
                      confidence_threshold: Optional[float] = None) -> pd.DataFrame:
        # TODO: write docstring.

        positive_data = features.loc[labels[labels == 1].index]
        if confidence_threshold:
            probs = self._model.predict_proba(
                positive_data.values, pos_label_only=True)
            positive_confident_df = pd.Series(
                probs.flatten(), index=positive_data.index)
            positive_confident_df = positive_confident_df[positive_confident_df >=
                                                          confidence_threshold]
        if len(positive_confident_df) == 0:
            raise ValueError(
                "Dataset is empty after excluding negative outcome examples."
            )
        return features.loc[positive_confident_df.index]

    def _cluster_data(self, data: pd.DataFrame, k_directions: int,
                      random_seed: Optional[int] = None) -> (pd.DataFrame, np.ndarray):
        # TODO: write docstring.

        km = KMeans(n_clusters=k_directions, random_state=random_seed)
        cluster_assignments = np.array(km.fit_predict(data)).reshape(-1, 1)
        cluster_assignments_df = pd.DataFrame(
            index=data.index,
            columns=["datapoint_cluster"],
            data=cluster_assignments
        )
        cluster_centers = km.cluster_centers_
        return cluster_assignments_df, cluster_centers

    def compute_all_directions(self, poi: pd.DataFrame, noise: float = None) -> pd.DataFrame:
        # TODO: write docstring.
        directions = []
        for cluster_index in range(self.k_directions):
            cluster_data = self.processed_data.loc[self.clusters_assignments[
                self.clusters_assignments["datapoint_cluster"] == cluster_index].index]
            direction = self.compute_direction(poi, cluster_data, noise)
            direction["cluster_index"] = cluster_index
            directions.append(direction)
        return pd.concat(directions)

    def compute_direction(self, poi: pd.DataFrame, cluster_data: pd.DataFrame, noise: float = None) -> pd.DataFrame:
        # TODO: Zero out immutable features here, allow changes to dist function. docstring
        # Feedback: the distance metric we use is a hyperparameter/decision choice.
        # As a future feature request, this should be changed to be passed as a parameter.
        immutable_feats = self._data_interface.get_processed_immutable_cat_feats()
        cluster_data_filtered = cluster_data.copy()
        cluster_data_filtered[immutable_feats] = 0
        poi[immutable_feats] = 0
        diff = cluster_data_filtered.values - poi.values
        dist = np.sqrt(np.power(diff, 2).sum(axis=1))
        alpha_val = self.volcano_alpha(dist)
        if np.isnan(alpha_val).any():
            raise RuntimeError(
                f"Alpha function returned NaN values: {alpha_val}")
        direction = diff.T @ alpha_val
        direction_df = poi.copy()
        
        direction_df.iloc[0] = pd.Series(direction)
        
        if self.directions_rescaler == "normalize":
            direction_df = direction_df/len(cluster_data)
        elif self.directions_rescaler == "constant step size":
            direction_df = self.constant_step_size(
                direction_df, self.step_size)
        
        direction_df[direction_df[self._data_interface.unidirection_features[0]] > 0] = 0
        direction_df[direction_df[self._data_interface.unidirection_features[1]] < 0] = 0
        if self._data_interface.ordinal_features:
            if self._data_interface.get_scaler():
                cols_to_scale = self._data_interface.ordinal_features
                for c in cols_to_scale:
                    direction_df[c] = self._data_interface.inverse_scale_ordinal(c,direction_df)
                """data_interface_scaler = self._data_interface.get_scaler()
                cols_to_scale = self._data_interface.get_scaled_features()
                scaled_direction_df = direction_df.copy()[cols_to_scale]
                scaled_direction_df[cols_to_scale] = data_interface_scaler.inverse_transform(
                    scaled_direction_df)
                scaled_direction_df[self._data_interface.ordinal_features] = scaled_direction_df[self._data_interface.ordinal_features].round()
                direction_df[cols_to_scale] = data_interface_scaler.transform(scaled_direction_df)"""
            else:
                direction_df[self._data_interface.ordinal_features] = direction_df[self._data_interface.ordinal_features].round()
        
        #noise for experiments
        if noise:
            direction_df = self.randomly_perturb_direction(direction_df,noise,immutable_feats)

        return direction_df

    def compute_all_paths(self, poi: pd.DataFrame, noise: float = None) -> list:
        # TODO: write docstring.
        # TODO: please consider parallelizing this / avoiding looping over the directions
        # on a per-PoI basis. This will be the big computational bottleneck.
        def compute_path(d, poi):
            cluster_index = d["cluster_index"]
            d = d.drop(labels="cluster_index")
            new_poi = poi.copy()
            path = [new_poi]
            drct = d.to_frame().T
            for i in range(self.max_iterations):
                new_poi = new_poi.add(drct, fill_value=0)

                if self._data_interface.categorical_features:
                    for feat_name, one_hot_feat_names in self._data_interface.get_encoded_categorical_feats().items():
                        one_hot_feats_constrained = np.zeros_like(
                            new_poi[one_hot_feat_names].values)
                        one_hot_feats_constrained[np.arange(len(new_poi[one_hot_feat_names])),
                                                  new_poi[one_hot_feat_names].values.argmax(1)] = 1
                        new_poi[one_hot_feat_names] = one_hot_feats_constrained
                #do we need to cap ordinal variables?
                path.append(new_poi)
                if self._model.predict_proba(new_poi, pos_label_only=True)[0][0] >= self._confidence_threshold:
                    break
                cluster_data = self.processed_data.loc[self.clusters_assignments[
                    self.clusters_assignments["datapoint_cluster"] == cluster_index].index]
                drct = self.compute_direction(new_poi, cluster_data, noise)
            return path
        directions = self.compute_all_directions(poi, noise)
        directions["path"] = directions.apply(
            lambda d: compute_path(d, poi), axis=1)
        return directions["path"].values
    
    def get_clusters(self):
        return self.clusters_assignments, self.cluster_centers

    def volcano_alpha(self, dist: np.ndarray, cutoff=0.5, degree=2) -> np.ndarray:
        return 1 / np.where(dist <= cutoff, cutoff, dist) ** degree

    def constant_step_size(self, direction: pd.DataFrame, step_size: float = 1) -> pd.DataFrame:
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

    def randomly_perturb_direction(self, direction: pd.DataFrame, ratio: float, immutable_feats) -> pd.DataFrame:
        """Randomly changes a vector's direction while maintaining its magnitude.

        The amount of random perturbation is determined by the `ratio` argument.
        If ratio=0.5, then random noise with magnitude 50% of the original
        direction is added. If ratio=1, then random noise with magnitude equal to
        the original direction is added. The direction is always rescaled to have
        its original magnitude after adding the random noise.

        Args:
            direction: The vector to perturb.
            ratio: The amount of random noise to add as a ratio of the direction's
                original magnitude.
            random_generator: An optional random generator to use when perturbing
                the direction. Otherwise defaults to np.random.normal().

        Returns:
            A new vector of equal magnitude to the original but with a randomly
            perturbed direction.
        """
        new_direction = direction.copy()
        new_direction[immutable_feats] = 0
        new_direction[self._data_interface.categorical_features] = 0
        new_direction[self._data_interface.ordinal_features] = 0
        # Check for zeroes to avoid division by zero.
        direction_norm = np.linalg.norm(new_direction)
        if direction_norm == 0:
            return new_direction
        noise = np.random.normal(0, 1, len(new_direction))
        noise_norm = np.linalg.norm(noise)
        if noise_norm == 0:
            return new_direction

        noise = (noise / noise_norm) * ratio * direction_norm
        new_direction = new_direction + noise
        # Normalize noised direction and rescale to the original direction length.
        new_direction = (new_direction / np.linalg.norm(new_direction)
                         ) * direction_norm
        return new_direction

class StEPRecourse(RecourseInterface):
    """
    TODO: docstring
    """

    def __init__(self, model: model_interface.ModelInterface, data_interface: data_interface.DataInterface,
                 num_clusters: int, max_iterations: int, use_train_data: bool = True,
                 confidence_threshold: Optional[float] = None, random_seed: Optional[int] = None, step_size=None,
                 directions_rescaler="normalize", special_cluster_data = None
                 ) -> None:

        self.StEP_instance = StEP(num_clusters, data_interface, model, max_iterations, use_train_data,
                                  confidence_threshold, random_seed, step_size, directions_rescaler,special_cluster_data)

    def get_counterfactuals(self, poi: pd.DataFrame, noise: float = None) -> Sequence:
        # TODO: write docstring.
        cfs = []
        paths = self.StEP_instance.compute_all_paths(poi, noise)
        for p in paths:
            cfs.append(p[-1])
        return cfs

    def get_paths(self, poi: pd.DataFrame, noise: float = None) -> Sequence:
        # TODO: write docstring.
        return self.StEP_instance.compute_all_paths(poi, noise)
    
    def get_counterfactuals_from_paths(self, paths: Sequence) -> Sequence:
        # TODO: write docstring.
        cfs = []
        for p in paths:
            cfs.append(p[-1])
        return cfs
    
    def get_clusters(self):
        return self.StEP_instance.get_clusters()
