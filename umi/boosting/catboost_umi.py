from catboost import CatBoost, CatBoostClassifier, CatBoostRegressor
from pandas import DataFrame

from umi import Objective
from umi.boosting.boosting_umi import BoostingUMI
from utils import names_to_id


class CatboostUMI(BoostingUMI):
    model: CatBoost

    def _get_model_from_objective(self, objective: Objective, model_args: dict) -> CatBoost:
        if objective == Objective.CLASSIFICATION:
            return CatBoostClassifier(**model_args)
        elif objective == Objective.REGRESSION:
            return CatBoostRegressor(**model_args)
        else:
            raise NotImplementedError("Unknown objective")

    def fit(self, x_train, y_train, x_val=None, y_val=None, **kwargs):
        if 'cat_features' not in kwargs and isinstance(x_train, DataFrame) and self.cat_features is not None:
            if len(self.cat_features) > 0 and isinstance(self.cat_features[0], str):
                kwargs['cat_features'] = names_to_id(x_train, self.cat_features)
            else:
                kwargs['cat_features'] = self.cat_features
        super().fit(x_train, y_train, x_val, y_val, **kwargs)
