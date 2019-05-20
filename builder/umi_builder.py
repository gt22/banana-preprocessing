from typing import Dict, Callable, Type, Optional, List, Union

from umi.base_umi import Objective, UnifiedModelInterface as UMI
from umi.sklearn_models import known_models as known_sklearn_models
from umi.sklearn_umi import SklearnUMI
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
    'dense': build_mlp
}

for n, m in known_sklearn_models.items():
    builder_map[n] = create_sklearn_builder(m)
