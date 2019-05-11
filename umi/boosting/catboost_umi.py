from catboost import CatBoost, CatBoostClassifier, CatBoostRegressor
from umi import Objective
from umi.boosting.boosting_umi import BoostingUMI


class CatboostUMI(BoostingUMI):
    model: CatBoost

    def _get_model_from_objective(self, objective: Objective, model_args: dict) -> CatBoost:
        if objective == Objective.CLASSIFICATION:
            return CatBoostClassifier(**model_args)
        elif objective == Objective.REGRESSION:
            return CatBoostRegressor(**model_args)
        else:
            raise NotImplementedError("Unknown objective")
