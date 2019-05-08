from umi.base_umi import Objective, UnifiedModelInterface


def build_catboost(cfg: dict, objective: Objective, name: str) -> UnifiedModelInterface:
    from umi.boosting.catboost_umi import CatboostUMI
    return CatboostUMI(objective, name, **cfg)


def build_lgbm(cfg: dict, objective: Objective, name: str) -> UnifiedModelInterface:
    from umi.boosting.lgbm_umi import LgbmUMI
    return LgbmUMI(objective, name, **cfg)


def build_xgb(cfg: dict, objective: Objective, name: str) -> UnifiedModelInterface:
    from umi.boosting.xgb_umi import XgbUMI
    return XgbUMI(objective, name, **cfg)


def build_mlp(cfg: dict, objective: Objective, name: str) -> UnifiedModelInterface:
    from umi.neuro.mlp_umi import MlpUmi
    return MlpUmi(objective, name, **cfg)


builder_map = {
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
