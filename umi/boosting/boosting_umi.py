from typing import Optional, List, Union

from umi import Objective
from umi.sklearn_umi import SklearnUMI


class BoostingUMI(SklearnUMI):

    def __init__(self, objective: Objective, model_name: str, class_num: Optional[int] = None,
                 cat_features: Optional[Union[List[str], List[int]]] = None, **kwargs):
        super().__init__(objective, model_name, class_num, cat_features, **kwargs)

    # For some reason, python gives warning when superclass implementation is just to throw NIE, so here is an override
    def _initialize_model(self, **kwargs):
        raise NotImplementedError("Model initialization is not defined for this class, use one of the subclasses")

    def fit(self, x_train, y_train, x_val=None, y_val=None, **kwargs):
        return super().fit(x_train, y_train, eval_set=(x_val, y_val) if x_val is not None else None)
