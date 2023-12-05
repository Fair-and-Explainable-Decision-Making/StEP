from ..models.model_interface import ModelInterface
from ..data.data_interface import DataInterface
import dice_ml
import pandas as pd
from recourse_interface import RecourseInterface


class DiceRecourse(RecourseInterface):
    def __init__(self, model: ModelInterface, data_interface: DataInterface, backend: str = "sklearn") -> None:
        self._model = model
        self._data_interface = data_interface
        train_dataset, test_dataset, _, _ = data_interface.get_split_data() 
        self._dice_data = dice_ml.Data(
            dataframe=train_dataset,
            continuous_features=data_interface.continuous_features(),
            outcome_name=data_interface.target_feature,
        )
        self._dice_model = dice_ml.Model(model=self._model, backend=backend)
        if backend == "PYT":
            self._dice = dice_ml.Dice(self._dice_data, self._dice_model, method="gradient")
        else:
            self._dice = dice_ml.Dice(self._dice_data, self._dice_model, method="random")
    
    def get_counterfactuals(self, poi: pd.DataFrame, num_CFs: int = 1, sparsity_param: float = 0.1):
        dice_exp = self._dice.generate_counterfactuals(
            poi,
            total_CFs=num_CFs,
            desired_class="opposite",
            posthoc_sparsity_param=sparsity_param,
        )
        return dice_exp.cf_examples_list[0].final_cfs_df

    def get_paths(self):
        pass