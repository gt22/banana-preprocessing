from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional
import os


class Objective(Enum):
    REGRESSION = 'regression'
    CLASSIFICATION = 'classification'


class UnifiedModelInterface(ABC):
    objective: Objective
    model_name: str
    class_num = Optional[int]

    def __init__(self, objective: Objective, model_name: str, class_num: Optional[int] = None):
        if objective != Objective.CLASSIFICATION and class_num is not None:
            raise ValueError("class_num should only be used with objective=CLASSIFICATION")
        self.objective = objective
        self.model_name = model_name
        self.class_num = class_num

    @abstractmethod
    def fit(self, x_train, y_train, x_val, y_val, **kwargs):
        pass

    @abstractmethod
    def predict(self, x, **kwargs):
        pass

    @abstractmethod
    def predict_proba(self, x, **kwargs):
        pass

    @abstractmethod
    def save(self, fold_dir, **kwargs):
        if not os.path.exists(fold_dir):
            os.makedirs(fold_dir)

    @abstractmethod
    def on_train_end(self, **kwargs):
        pass
