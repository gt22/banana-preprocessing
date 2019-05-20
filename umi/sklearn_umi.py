import os
import pickle
from typing import Optional, Any, List, Union

from umi import UnifiedModelInterface, Objective


class SklearnUMI(UnifiedModelInterface):
    model: Any

    def __init__(self, objective: Objective, model_name: str, class_num: Optional[int] = None,
                 cat_features: Optional[Union[List[str], List[int]]] = None, **kwargs):
        super().__init__(objective, model_name, class_num, cat_features, **kwargs)

    def _initialize_model(self, **kwargs):
        raise NotImplementedError("Model initialization is not defined for this class, use one of the subclasses")

    def fit(self, x_train, y_train, x_val=None, y_val=None, **kwargs):
        if hasattr(self.model, 'fit'):
            return self.model.fit(x_train, y_train, **kwargs)
        else:
            raise NotImplementedError(f"Couldn't find 'fit' method on {self.model}, are you sure this is a model?")

    def predict(self, x, **kwargs):
        if hasattr(self.model, 'predict'):
            return self.model.predict(x, **kwargs)
        else:
            raise NotImplementedError(f"Couldn't find 'predict' method on {self.model}, are you sure this is a model?")

    def predict_proba(self, x, **kwargs):
        if hasattr(self.model, 'predict_proba'):
            pred = self.model.predict_proba(x, **kwargs)
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
