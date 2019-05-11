from umi import UnifiedModelInterface, Objective
from typing import Optional, Any
import pickle
import os


class SklearnUMI(UnifiedModelInterface):
    model: Any

    def __init__(self, model, model_name: str, class_num: Optional[int] = None,
                 objective: Optional[Objective] = None):
        if objective is None:
            objective = self._get_objective_from_model(model)
        super().__init__(objective, model_name, class_num)
        self.model = model

    @staticmethod
    def _get_objective_from_model(model):
        if hasattr(model, '_estimator_type'):
            # noinspection PyProtectedMember
            et = model._estimator_type
            if et == 'classifier':
                return Objective.CLASSIFICATION
            elif et == 'regressor':
                return Objective.REGRESSION
            else:
                raise NotImplementedError(f"Unknown _estimator_type '{et}', likely this model won't work with UMI."
                                          f" Specify objective explicitly to try anyway.")
        else:
            raise ValueError(f"Couldn't find '_estimator_type' for {model}, please specify objective explicitly")

    def fit(self, x_train, y_train, x_val=None, y_val=None, **kwargs):
        if hasattr(self.model, 'fit'):
            return self.model.fit(x_train, y_train)
        else:
            raise NotImplementedError(f"Couldn't find 'fit' method on {self.model}, are you sure this is a model?")

    def predict(self, x, **kwargs):
        if hasattr(self.model, 'predict'):
            return self.model.predict(x)
        else:
            raise NotImplementedError(f"Couldn't find 'predict' method on {self.model}, are you sure this is a model?")

    def predict_proba(self, x, **kwargs):
        if hasattr(self.model, 'predict_proba'):
            pred = self.model.predict_proba(x)
            if self.objective == Objective.CLASSIFICATION and pred.shape[1] == 2:
                pred = pred[:, 1]
            return pred
        else:
            raise NotImplementedError(f"Couldn't find 'predict_proba' method on {self.model}, " +
                                      f"are you sure this is classification model?"
                                      if self.objective == Objective.CLASSIFICATION
                                      else f"this method is usually only present for classification models")

    def save(self, fold_dir, **kwargs):
        super().save(fold_dir)
        with open(os.path.join(fold_dir, f'{self.model_name}.pickle'), 'wb') as f:
            pickle.dump(self.model, f)

    def on_train_end(self):
        del self.model
