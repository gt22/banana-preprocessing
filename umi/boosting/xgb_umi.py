from xgboost import XGBModel, XGBClassifier, XGBRegressor

from umi import Objective
from umi.boosting.boosting_umi import BoostingUMI


class XgbUMI(BoostingUMI):

    model: XGBModel

    def _initialize_model(self, **kwargs):
        if self.objective == Objective.CLASSIFICATION:
            self.model = XGBClassifier(**kwargs)
        elif self.objective == Objective.REGRESSION:
            self.model = XGBRegressor(**kwargs)
        else:
            raise NotImplementedError("Unknown objective")
