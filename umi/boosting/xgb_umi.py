from xgboost import XGBModel, XGBClassifier, XGBRegressor
from umi import Objective
from umi.boosting.boosting_umi import BoostingUMI


class XgbUMI(BoostingUMI):

    model: XGBModel

    def _get_model_from_objective(self, objective: Objective, model_args: dict) -> XGBModel:
        if objective == Objective.CLASSIFICATION:
            return XGBClassifier(**model_args)
        elif objective == Objective.REGRESSION:
            return XGBRegressor(**model_args)
        else:
            raise NotImplementedError("Unknown objective")
