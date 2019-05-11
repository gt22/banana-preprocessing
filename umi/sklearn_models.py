from __future__ import annotations
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR
from umi.sklearn_umi import SklearnUMI
from umi import Objective
from typing import Dict, Type, Optional


class ModelClass(SklearnUMI):

    def __init__(self, objective: Objective, model_name: str, class_num: Optional[int] = None, **kwargs):
        super().__init__(self.create_model(objective, **kwargs),
                         model_name, class_num, objective)

    def create_model(self, objective: Objective, **kwargs):
        raise NotImplementedError()


known_models: Dict[str, Type[ModelClass]] = {}


def create_model_class(name: str, reg_model, cls_model) -> Type[ModelClass]:
    class SubModelClass(ModelClass):

        def create_model(self, objective: Objective, **kwargs):
            return (reg_model if objective == Objective.REGRESSION else cls_model)(**kwargs)

    known_models[name] = SubModelClass
    return SubModelClass


LinearModelUMI = create_model_class('linear', LinearRegression, LogisticRegression)
RandomForestUMI = create_model_class('rf', RandomForestRegressor, RandomForestClassifier)
SvmUMI = create_model_class('svm', SVR, SVC)
LinearSvmUMI = create_model_class('linearsvm', LinearSVR, LinearSVC)
