from lightgbm import LGBMModel, LGBMClassifier, LGBMRegressor
from umi.base_umi import Objective
from umi.boosting.boosting_umi import BoostingUMI


class LgbUMI(BoostingUMI):
    model: LGBMModel

    def _get_model_from_objective(self, objective: Objective, model_args: dict) -> LGBMModel:
        if objective == Objective.CLASSIFICATION:
            return LGBMClassifier(**model_args)
        elif objective == Objective.REGRESSION:
            return LGBMRegressor(**model_args)
        else:
            raise NotImplementedError("Unknown objective")
