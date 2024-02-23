from models import model_interface
from data import data_interface
import dice_ml
import pandas as pd
from recourse.recourse_interface import RecourseInterface
from typing import Optional, Sequence
from raiutils.exceptions import UserConfigValidationException


class DiceRecourse(RecourseInterface):
    def __init__(self, model: model_interface.ModelInterface, data_interface: data_interface.DataInterface, 
                 backend: str = "sklearn", default_k=1, default_sparsity=0.1, confidence_threshold: float = 0.5,
                 random_seed: Optional[int] = None) -> None:
        """
        TODO: docstring
        """
        self._model = model
        self._data_interface = data_interface
        features, _, labels, _ = data_interface.get_train_test_split()
        dataset = pd.concat([features, labels], axis=1)
        self._dice_data = dice_ml.Data(
            dataframe=dataset,
            continuous_features=data_interface.continuous_features,
            outcome_name=data_interface.target_feature,
        )
        self._dice_model = dice_ml.Model(model=self._model, backend=backend)
        # TODO: better naming for PYT
        if backend == "PYT":
            self._dice = dice_ml.Dice(
                self._dice_data, self._dice_model, method="gradient")
        else:
            self._dice = dice_ml.Dice(
                self._dice_data, self._dice_model, method="random")
        self._backend = backend
        self._default_k = default_k
        self._default_sparsity = default_sparsity
        self._orig_data_types = features.dtypes
        self._confidence_threshold = confidence_threshold
        self._random_seed = random_seed

    def get_counterfactuals(self, poi: pd.DataFrame, sparsity_param: float = None, num_CFs: int = None, confidence_threshold = None):
        if not num_CFs:
            num_CFs = self._default_k
        if not sparsity_param:
            sparsity_param = self._default_sparsity
        if not confidence_threshold:
            confidence_threshold = self._confidence_threshold
        poi = poi.astype(self._orig_data_types)

        if self._backend != "PYT":
            dice_exp = self._dice.generate_counterfactuals(
                poi,
                total_CFs=num_CFs,
                desired_class="opposite",
                posthoc_sparsity_param=sparsity_param,
                stopping_threshold=confidence_threshold,
                random_seed = self._random_seed
            )
        else:
            dice_exp = self._dice.generate_counterfactuals(
                poi,
                total_CFs=num_CFs,
                desired_class="opposite",
                posthoc_sparsity_param=sparsity_param,
                stopping_threshold=confidence_threshold
            )
        cfs = dice_exp.cf_examples_list[0].final_cfs_df
        cfs = cfs.drop(columns=self._data_interface.target_feature)
        return cfs

    def get_paths(self, poi: pd.DataFrame, sparsity_param: float = None, num_CFs: int = None, confidence_threshold: float = None):
        try:
            cfs = self.get_counterfactuals(poi, sparsity_param, num_CFs, confidence_threshold)
            cfs = [cfs.loc[[i]] for i in cfs.index] 
        except UserConfigValidationException:
            return [[poi]]*self._default_k
        except ValueError:
            return [[poi]]*self._default_k
        except:
            raise
        paths = []
        for cf in cfs:
            paths.append([poi, cf])
        return paths

    def get_counterfactuals_from_paths(self, paths):
        # TODO: write docstring.
        cfs = []
        for p in paths:
            cfs.append(p[-1])
        return cfs
