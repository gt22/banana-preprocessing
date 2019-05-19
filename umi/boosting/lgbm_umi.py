from lightgbm import LGBMModel, LGBMClassifier, LGBMRegressor

from umi import Objective
from umi.boosting.boosting_umi import BoostingUMI


class LgbmUMI(BoostingUMI):
    model: LGBMModel

    def _get_model_from_objective(self, objective: Objective, model_args: dict) -> LGBMModel:
        if objective == Objective.CLASSIFICATION:
            return LGBMClassifier(**model_args)
        elif objective == Objective.REGRESSION:
            return LGBMRegressor(**model_args)
        else:
            raise NotImplementedError("Unknown objective")

    def fit(self, x_train, y_train, x_val=None, y_val=None, **kwargs):
        if 'categorical_feature' not in kwargs and self.cat_features is not None:
            kwargs['categorical_feature'] = self.cat_features
        super().fit(x_train, y_train, x_val, y_val, **kwargs)
