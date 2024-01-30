from models import model_interface
from data import data_interface
import dice_ml
import pandas as pd
from recourse_interface import RecourseInterface


class DiceRecourse(RecourseInterface):
    def __init__(self, model: model_interface.ModelInterface, data_interface: data_interface.DataInterface, backend: str = "sklearn") -> None:
        """
        TODO: docstring
        """
        self._model = model
        self._data_interface = data_interface
        train_dataset, test_dataset, _, _ = data_interface.get_train_test_split()
        self._dice_data = dice_ml.Data(
            dataframe=train_dataset,
            continuous_features=data_interface.continuous_features(),
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

    def get_counterfactuals(self, poi: pd.DataFrame, num_CFs: int = 1, sparsity_param: float = 0.1):
        dice_exp = self._dice.generate_counterfactuals(
            poi,
            total_CFs=num_CFs,
            desired_class="opposite",
            posthoc_sparsity_param=sparsity_param,
        )
        return dice_exp.cf_examples_list[0].final_cfs_df

    def get_paths(self, poi: pd.DataFrame, num_CFs: int = 1, sparsity_param: float = 0.1):
        cfs = self.get_counterfactuals(poi, num_CFs, sparsity_param)
        paths = []
        for cf in cfs:
            paths.append([poi, cf])
        return paths
