from umi import Objective
from umi.sklearn_umi import SklearnUMI
from typing import Optional


class BoostingUMI(SklearnUMI):

    def __init__(self, objective: Objective, model_name: str, class_num: Optional[int] = None, model=None, **kwargs):
        if model is None:
            model = self._get_model_from_objective(objective, kwargs)
        super().__init__(model, model_name, class_num, objective)

    def _get_model_from_objective(self, objective: Objective, model_args: dict):
        raise NotImplementedError("This class doesn't support inferring models. You should pass model into constructor")

    def fit(self, x_train, y_train, x_val=None, y_val=None, **kwargs):
        return super().fit(x_train, y_train, eval_set=(x_val, y_val) if x_val is not None else None)
