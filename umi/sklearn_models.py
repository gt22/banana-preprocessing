from __future__ import annotations

from typing import Dict, Type, Optional

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR

from umi import Objective
from umi.sklearn_umi import SklearnUMI

known_models: Dict[str, Type[SklearnUMI]] = {}


def create_model_class(name: str, reg_model, cls_model) -> Type[SklearnUMI]:
    class SubModelClass(SklearnUMI):

        def _initialize_model(self, **kwargs):
            self.model = (reg_model if self.objective == Objective.REGRESSION else cls_model)(**kwargs)

    known_models[name] = SubModelClass
    return SubModelClass


LinearModelUMI = create_model_class('linear', LinearRegression, LogisticRegression)
RandomForestUMI = create_model_class('rf', RandomForestRegressor, RandomForestClassifier)
SvmUMI = create_model_class('svm', SVR, SVC)
LinearSvmUMI = create_model_class('linearsvm', LinearSVR, LinearSVC)
