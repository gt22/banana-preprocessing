from umi.base_umi import Objective, UnifiedModelInterface
from typing import Dict, Callable
from umi.sklearn_models import known_models as known_sklearn_models

UMIBuilder = Callable[[dict, Objective, str, int], UnifiedModelInterface]


def build_catboost(cfg: dict, objective: Objective, name: str, class_num: int) -> UnifiedModelInterface:
    from umi.boosting.catboost_umi import CatboostUMI
    return CatboostUMI(objective, name, class_num, **cfg)


def build_lgbm(cfg: dict, objective: Objective, name: str, class_num: int) -> UnifiedModelInterface:
    from umi.boosting.lgbm_umi import LgbmUMI
    return LgbmUMI(objective, name, class_num, **cfg)


def build_xgb(cfg: dict, objective: Objective, name: str, class_num: int) -> UnifiedModelInterface:
    from umi.boosting.xgb_umi import XgbUMI
    return XgbUMI(objective, name, class_num, **cfg)


def build_mlp(cfg: dict, objective: Objective, name: str, class_num: int) -> UnifiedModelInterface:
    from umi.neuro.mlp_umi import MlpUmi
    return MlpUmi(objective, name, class_num=class_num, **cfg)


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
    builder_map[n] = m.builder
