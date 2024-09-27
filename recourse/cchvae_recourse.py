from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
from numpy import linalg as LA
from models import model_interface
from data import data_interface
from recourse.recourse_interface import RecourseInterface
from data.data_interface import DataInterface
from typing import Optional, Sequence
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

#from carla import log
#from carla.models.api import MLModel

from recourse.utils.vae import VariationalAutoencoder
from recourse.utils.carla_utils import (
    merge_default_parameters,
    reconstruct_encoding_constraints,
)

def constrain_one_hot_cat_features(poi: pd.DataFrame, data_interface: DataInterface):
    if isinstance(poi,pd.Series):
        poi = poi.to_frame().T 
    for feat_name, one_hot_feat_names in data_interface.get_encoded_categorical_feats().items():
        one_hot_feats_constrained = np.zeros_like(
            poi[one_hot_feat_names].values)
        one_hot_feats_constrained[np.arange(len(poi[one_hot_feat_names])),
                                    poi[one_hot_feat_names].values.argmax(1)] = 1
        poi[one_hot_feat_names] = one_hot_feats_constrained
    return poi

class CCHVAE():
    """
    Implementation of CCHVAE [1]_

    Parameters
    ----------
    mlmodel : carla.model.MLModel
        Black-Box-Model
    hyperparams : dict
        Dictionary containing hyperparameters. See Notes below to see its content.

    Methods
    -------
    get_counterfactuals:
        Generate counterfactual examples for given factuals.

    Notes
    -----
    - Hyperparams
        Hyperparameter contains important information for the recourse method to initialize.
        Please make sure to pass all values as dict with the following keys.

        * "data_name": str
            name of the dataset
        * "n_search_samples": int, default: 300
            Number of generated candidate counterfactuals.
        * "p_norm": {1, 2}
            Defines L_p norm for distance calculation.
        * "step": float, default: 0.1
            Step size for each generated candidate counterfactual.
        * "max_iter": int, default: 1000
            Number of iterations per factual instance.
        * "clamp": bool, default: True
            Feature values will be clamped between 0 and 1
        * "binary_cat_features": bool, default: True
            If true, the encoding of x is done by drop_if_binary.
        * "vae_params": Dict
            With parameter for VAE.

            + "layers": list
                List with number of neurons per layer.
            + "train": bool, default: True
                Decides if a new Autoencoder will be learned.
            + "lambda_reg": float, default: 1e-6
                Hyperparameter for variational autoencoder.
            + "epochs": int, default: 5
                Number of epochs to train VAE
            + "lr": float, default: 1e-3
                Learning rate for VAE training
            + "batch_size": int, default: 32
                Batch-size for VAE training

    .. [1] Pawelczyk, Martin, Klaus Broelemann and Gjergji Kasneci. “Learning Model-Agnostic Counterfactual Explanations
          for Tabular Data.” Proceedings of The Web Conference 2020 (2020): n. pag..
    """

    _DEFAULT_HYPERPARAMS = {
        "data_name": None,
        "n_search_samples": 1000,
        "p_norm": 2,
        "step": 1.0,
        "max_iter": 1000,
        "clamp": True,
        "binary_cat_features": True,
        "confidence_threshold": 0.5,
        "k_directions" : 1,
        "vae_params": {
            "layers": [ 512, 256, 8 ],
            "train": True,
            "kl_weight": 0.3,
            "lambda_reg": 1e-6,
            "epochs": 5,
            "lr": 1e-3,
            "batch_size": 32,
        },
        
    }

    def __init__(self, mlmodel: model_interface.ModelInterface, data_interface: data_interface.DataInterface, hyperparams: Dict = None) -> None:
        self._mlmodel = mlmodel
        self._data_interface = data_interface
        features, _, labels, _ = data_interface.get_train_test_split()
        self._mutable_mask = [0 if ele in self._data_interface.immutable_features 
               or ele in self._data_interface.get_processed_immutable_feats() else 1 
               for ele in features.columns.values]
        
        self._params = merge_default_parameters(hyperparams, self._DEFAULT_HYPERPARAMS)

        self._n_search_samples = self._params["n_search_samples"]
        self._p_norm = self._params["p_norm"]
        self._step = self._params["step"]
        self._max_iter = self._params["max_iter"]
        self._clamp = self._params["clamp"]
        self._k_directions = self._params["k_directions"]
        self._params["vae_params"]["layers"] = [sum(self._mutable_mask)] + self._params["vae_params"]["layers"]
        vae_params = self._params["vae_params"]
        self._scaler = MinMaxScaler().fit(features)
        features[features.columns] = self._scaler.transform(features)
        self._generative_model = self._load_vae(
            features, vae_params, self._mlmodel, self._params["data_name"]
        )
        
        self._feature_columns = features.columns.values
        self._confidence_threshold = self._params["confidence_threshold"]

    def _load_vae(
        self, data: pd.DataFrame, vae_params: Dict, mlmodel: model_interface.ModelInterface, data_name: str
    ) -> VariationalAutoencoder:
        
        
        generative_model = VariationalAutoencoder(
            data_name, vae_params["layers"], np.array(list(map(bool,self._mutable_mask)))
        )

        if vae_params["train"]:
            generative_model.fit(
                xtrain=data,
                kl_weight=vae_params["kl_weight"],
                lambda_reg=vae_params["lambda_reg"],
                epochs=vae_params["epochs"],
                lr=vae_params["lr"],
                batch_size=vae_params["batch_size"],
            )
        else:
            try:
                generative_model.load()
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    "Loading of Autoencoder failed. {}".format(str(exc))
                )

        return generative_model

    def _hyper_sphere_coordindates(
        self, instance, high: int, low: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param n_search_samples: int > 0
        :param instance: numpy input point array
        :param high: float>= 0, h>l; upper bound
        :param low: float>= 0, l<h; lower bound
        :param p: float>= 1; norm
        :return: candidate counterfactuals & distances
        """
        delta_instance = np.random.randn(self._n_search_samples, instance.shape[1])
        dist = (
            np.random.rand(self._n_search_samples) * (high - low) + low
        )  # length range [l, h)
        norm_p = LA.norm(delta_instance, ord=self._p_norm, axis=1)
        d_norm = np.divide(dist, norm_p).reshape(-1, 1)  # rescale/normalize factor
        delta_instance = np.multiply(delta_instance, d_norm)
        candidate_counterfactuals = instance + delta_instance
        return candidate_counterfactuals, dist

    def counterfactual_search(
        self, step: int, factual: pd.DataFrame
    ) -> pd.DataFrame:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # init step size for growing the sphere
        low = 0
        high = step
        # counter
        count = 0
        counter_step = 1
        tranformed_factual = self._scaler.transform(factual)
        torch_fact = torch.from_numpy(tranformed_factual).to(device)

        # get predicted label of instance
        """instance_label = np.argmax(
            self._mlmodel.predict_proba(torch_fact.float()).cpu().detach().numpy(),
            axis=1,
        )"""
        instance_label = self._mlmodel.predict(factual, confidence_threshold = self._confidence_threshold)

        # vectorize z
        z = self._generative_model.encode(
            torch_fact[:, self._generative_model.mutable_mask].float()
        )[0]
        # add the immutable features to the latents
        z = torch.cat([z, torch_fact[:, ~self._generative_model.mutable_mask]], dim=-1)
        z = z.cpu().detach().numpy()
        z_rep = np.repeat(z.reshape(1, -1), self._n_search_samples, axis=0)

        # make copy such that we later easily combine the immutables and the reconstructed mutables
        fact_rep = torch_fact.reshape(1, -1).repeat_interleave(
            self._n_search_samples, dim=0
        )

        candidate_dist: List = []
        x_ce: Union[np.ndarray, torch.Tensor] = np.array([])
        while True: #count <= self._max_iter or len(candidate_dist) <= 0:
            count = count + counter_step

            # STEP 1 -- SAMPLE POINTS on hyper sphere around instance
            latent_neighbourhood, _ = self._hyper_sphere_coordindates(z_rep, high, low)
            torch_latent_neighbourhood = (
                torch.from_numpy(latent_neighbourhood).to(device).float()
            )
            x_ce = self._generative_model.decode(torch_latent_neighbourhood)

            # add the immutable features to the reconstruction
            temp = fact_rep.clone()
            temp[:, self._generative_model.mutable_mask] = x_ce.double()
            x_ce = temp

            """x_ce = reconstruct_encoding_constraints(
                x_ce, cat_features_indices, self._params["binary_cat_features"]
            )"""
            x_ce = x_ce.detach().cpu().numpy()
            x_ce_df = pd.DataFrame(x_ce, columns = self._feature_columns)
            x_ce = pd.concat(x_ce_df.apply(lambda x: constrain_one_hot_cat_features(x, self._data_interface),axis=1).values).values
            x_ce = x_ce.clip(0, 1) if self._clamp else x_ce

            # STEP 2 -- COMPUTE l1 & l2 norms
            if self._p_norm == 1:
                distances = np.abs((x_ce - torch_fact.cpu().detach().numpy())).sum(
                    axis=1
                )
            elif self._p_norm == 2:
                distances = LA.norm(x_ce - torch_fact.cpu().detach().numpy(), axis=1)
            else:
                raise ValueError("Possible values for p_norm are 1 or 2")

            # counterfactual labels
            x_ce_untransformed = self._scaler.inverse_transform(x_ce)
            y_candidate = self._mlmodel.predict(x_ce_untransformed, confidence_threshold = self._confidence_threshold)
            """np.argmax(
                self._mlmodel.predict_proba(torch.from_numpy(x_ce).float())
                .cpu()
                .detach()
                .numpy(),
                axis=1,
            )"""
            indices = np.where(y_candidate != instance_label)
            candidate_counterfactuals = x_ce_untransformed[indices]
            candidate_dist = distances[indices]
            # no candidate found & push search range outside
            if len(candidate_dist) < self._k_directions:
                low = high
                high = low + step
            elif len(candidate_dist) >= self._k_directions:
                # certain candidates generated
                min_index = np.argsort(candidate_dist)[:self._k_directions]
                print("Counterfactual examples found")
                return candidate_counterfactuals[min_index]
            if int(count) > int(self._max_iter):
                print("k counterfactual examples not found")
                min_index = np.argsort(candidate_dist)[:self._k_directions]
                num_missing_cfs = self._k_directions - len(candidate_dist)
                filler_missing_cfs = np.repeat(factual.values, num_missing_cfs, axis=0)
                return np.concatenate((filler_missing_cfs, candidate_counterfactuals[min_index]), axis=0)

    def get_counterfactuals(self, poi: pd.DataFrame) -> pd.DataFrame:
        cfs_np = self.counterfactual_search(self._step,poi)
        return pd.DataFrame(cfs_np, columns = self._feature_columns)

class CCHVAERecourse(RecourseInterface):

    def __init__(self, model: model_interface.ModelInterface, data_interface: data_interface.DataInterface,
                k_directions: int, confidence_threshold: Optional[float] = None, 
                random_seed = 0, train_vae = True, max_iterations: int = 50
                ) -> None:
        tf.random.set_seed(random_seed)
        data_name = (data_interface.name+"_"+str(random_seed)).replace(" ", "")
        hyperparams = {
            "data_name": data_name,
            "confidence_threshold": confidence_threshold,
            "k_directions": k_directions,
            "max_iter": max_iterations,
            "vae_params": {
                "train": train_vae,
            } 
        }
        self._CCHVAE_instance = CCHVAE(model, data_interface, hyperparams)
        

    def get_counterfactuals(self, poi: pd.DataFrame) -> Sequence:
        # TODO: write docstring.
        cfs = self._CCHVAE_instance.get_counterfactuals(poi)
        return cfs

    def get_paths(self, poi: pd.DataFrame) -> Sequence:
        # TODO: write docstring.
        cfs = self.get_counterfactuals(poi)
        cfs = [cfs.loc[[i]] for i in cfs.index] 
        paths = []
        for cf in cfs:
            paths.append([poi, cf])
        return paths
    
    def get_counterfactuals_from_paths(self, paths: Sequence) -> Sequence:
        # TODO: write docstring.
        cfs = []
        for p in paths:
            cfs.append(p[-1])
        return cfs