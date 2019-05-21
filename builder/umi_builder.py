from typing import Dict, Callable, Type, Optional, List, Union

from umi.base_umi import Objective, UnifiedModelInterface as UMI
from umi.sklearn_models import known_models as known_sklearn_models
from umi.sklearn_umi import SklearnUMI
from collections import defaultdict
CatFeatures = Optional[Union[List[str], List[int]]]
UMIBuilder = Callable[[dict, Objective, str, int, CatFeatures], UMI]


def build_catboost(cfg: dict, objective: Objective, name: str, class_num: int, cf: CatFeatures) -> UMI:
    from umi.boosting.catboost_umi import CatboostUMI
    return CatboostUMI(objective, name, class_num, cf, **cfg)


def build_lgbm(cfg: dict, objective: Objective, name: str, class_num: int, cf: CatFeatures) -> UMI:
    from umi.boosting.lgbm_umi import LgbmUMI
    return LgbmUMI(objective, name, class_num, cf, **cfg)


def build_xgb(cfg: dict, objective: Objective, name: str, class_num: int, cf: CatFeatures) -> UMI:
    from umi.boosting.xgb_umi import XgbUMI
    return XgbUMI(objective, name, class_num, cf, **cfg)


def build_mlp(cfg: dict, objective: Objective, name: str, class_num: int, cf: CatFeatures) -> UMI:
    from umi.neuro.mlp_umi import MlpUmi
    return MlpUmi(objective, name, class_num=class_num, **cfg)


def build_ensemble(cfg: dict, objective: Objective, name: str, class_num: int, cf: CatFeatures) -> UMI:
    from umi.ensemble_umi import EnsembleUMI
    if 'models' not in cfg:
        raise ValueError("Ensemble model requires 'models' present in it's config")
    models = cfg.pop('models')
    if type(models) != list and type(models) != tuple:
        raise ValueError("Models for ensembling must be in list or tuple")
    ms = []
    for i, m in enumerate(models):
        if 'type' not in m:
            raise ValueError(f"Type not found in model {i}")
        t = m.pop('type')
        if t not in builder_map:
            raise ValueError(f"Unknown type {t} in model {i}")
        n = m.pop('name', t)
        n = f'{i}_{n}'
        ms.append(builder_map[t](m, objective, n, class_num, cf))
    return EnsembleUMI(ms, objective, name, class_num=class_num, cat_features=cf, **cfg)


def create_sklearn_builder(m: Type[SklearnUMI]) -> UMIBuilder:
    def sklearn_builder(cfg: dict, objective: Objective, name: str, class_num: int, cf: CatFeatures) -> UMI:
        return m(objective, name, class_num, **cfg)
    return sklearn_builder


builder_map: Dict[str, UMIBuilder] = {
    'catboost': build_catboost,
    'cb': build_catboost,
    'lightgbm': build_lgbm,
    'lgbm': build_lgbm,
    'lgb': build_lgbm,
    'xgboost': build_xgb,
    'xgb': build_xgb,
    'mlp': build_mlp,
    'dense': build_mlp,
    'ensemble': build_ensemble
}


def initialize_sklearn_builders():
    for n, m in known_sklearn_models.items():
        builder_map[n] = create_sklearn_builder(m)


initialize_sklearn_builders()
