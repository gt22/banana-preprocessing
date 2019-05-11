from __future__ import annotations
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR
from umi.sklearn_umi import SklearnUMI
from umi.base_umi import Objective
from typing import Dict, Type

known_models: Dict[str, Type[ModelClass]] = {}


def create_model_class(name: str, reg_model, cls_model):
    class ModelClass(SklearnUMI):

        def __init__(self, objective: Objective, model_name: str, **kwargs):
            super().__init__(reg_model(**kwargs) if objective == Objective.REGRESSION else cls_model(**kwargs),
                             model_name, objective=objective)

        @classmethod
        def builder(cls, cfg: dict, objective: Objective, model_name: str, class_num: int) -> ModelClass:
            return cls(objective, model_name, **cfg)
    known_models[name] = ModelClass
    return ModelClass


LinearModelUMI = create_model_class('linear', LinearRegression, LogisticRegression)
RandomForestUMI = create_model_class('rf', RandomForestRegressor, RandomForestClassifier)
SvmUMI = create_model_class('svm', SVR, SVC)
LinearSvmUMI = create_model_class('linearsvm', LinearSVR, LinearSVC)
