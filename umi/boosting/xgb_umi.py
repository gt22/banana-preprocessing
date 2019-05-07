from xgboost import XGBModel, XGBClassifier, XGBRegressor
from umi.base_umi import UnifiedModelInterface, Objective
from typing import Optional
import pickle
import os


class XgbUMI(UnifiedModelInterface):

    model: XGBModel

    def __init__(self, model: XGBModel, model_name: str, class_num: Optional[int] = None,
                 objective: Optional[Objective] = None):
        if objective is None:
            objective = self._get_objective_from_model(model)
        super().__init__(objective, model_name, class_num)
        self.model = model

    @staticmethod
    def _get_objective_from_model(model: XGBModel):
        if isinstance(model, XGBRegressor):
            return Objective.REGRESSION
        elif isinstance(model, XGBClassifier):
            return Objective.CLASSIFICATION
        else:
            raise ValueError(f"Unknown XGB model {model}, likely it won't work with UMI."
                             f" Specify objective explicitly to try anyway.")

    def fit(self, x_train, y_train, x_val, y_val, **kwargs):
        return self.model.fit(x_train, y_train, eval_set=(x_val, y_val) if x_val is not None else None, **kwargs)

    def predict(self, x, **kwargs):
        return self.model.predict(x, **kwargs)

    def predict_proba(self, x, **kwargs):
        if hasattr(self.model, 'predict_proba'):
            # noinspection PyUnresolvedReferences
            return self.model.predict_proba(x, **kwargs)
        else:
            if self.objective == Objective.CLASSIFICATION:
                raise ValueError(f"Couldn't find 'predict_proba' method on {self.model}, "
                                 f"are you sure this is classification model?")
            else:
                raise ValueError(f"Couldn't find 'predict_proba' method on {self.model}, "
                                 f"this method is usually only present for classification models")

    def save(self, fold_dir, **kwargs):
        super().save(fold_dir)
        pickle.dump(self.model, os.path.join(fold_dir, f'{self.model_name}.pickle'))

    def on_train_end(self):
        del self.model
