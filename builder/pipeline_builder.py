from umi.base_umi import Objective, UnifiedModelInterface
from preprocessing import Preprocessing, SplitterType, ScalerType
from typing import Optional
from builder.umi_builder import builder_map
from scorer.scorer import Scorer, SaveTactics
from pipeline import Pipeline

config_example = {
    'objective': 'classification',
    'name': 'cb_model',
    'preprocessing': {
        'scaler': 'minmax',
        'kfold': 3
    },
    'model': {
        'type': 'catboost',
        'iterations': 100
    },
    'scorer': {
        'metrics': 'accuracy',
        'save': 'iob',
        'criterion': 'all_improve',
        'save_dir': 'score_data'
    },
    'score_merger': 'max',
    'use_proba': False
}

DEFAULT_KFOLD = 5


def build_preprocessing(cfg: dict) -> Preprocessing:
    scaler = ScalerType(cfg.get('scaler', 'none'))
    splitter = SplitterType(cfg.get('splitter', 'kfold'))
    kfold = cfg.get('kfold', DEFAULT_KFOLD)
    return Preprocessing(scaler, splitter, kfold)


def build_model(cfg: dict, objective: Objective, class_num: Optional[int], name: str) -> UnifiedModelInterface:
    model_type = cfg['type']
    if model_type not in builder_map:
        raise ValueError(f"Unknown type '{model_type}'")
    model_cfg = cfg.copy()
    model_cfg.pop('type')
    model_cfg.pop('name', None)
    return builder_map[model_type](model_cfg, objective, name, class_num)


def build_scorer(cfg: dict, name: str) -> Scorer:
    if 'metrics' not in cfg:
        raise ValueError("'scorer.metrics' is required")
    metrics = cfg['metrics']
    if type(metrics) is str:
        metrics = [metrics]
    save_tactics = cfg.get('save', None)
    criterion = cfg.get('criterion', None)
    save_dir = cfg.get('save_dir', None)

    creation_cfg = {}
    if save_tactics is not None:
        creation_cfg['save_tactics'] = SaveTactics(save_tactics)
    if criterion is not None:
        creation_cfg['improvement_criterion'] = criterion
    if save_dir is not None:
        creation_cfg['save_dir'] = save_dir
    return Scorer(name, metrics, **creation_cfg)


def build_pipeline(cfg: dict):
    if 'objective' not in cfg:
        raise ValueError("'objective' is required")
    if 'model' not in cfg:
        raise ValueError("'model' is required")
    if 'type' not in cfg['model']:
        raise ValueError("'model.type' is required")

    obj: Objective = Objective(cfg['objective'])
    if obj == Objective.CLASSIFICATION and 'use_proba' not in cfg:
        raise ValueError("For classification task, 'use_proba' is required")

    name = cfg.get('name', cfg['model']['type'])
    class_num: Optional[int] = cfg.get('class_num', 2) if obj == Objective.CLASSIFICATION else None

    pipeline_cfg = {
        'preproc': build_preprocessing(cfg['preprocessing']) if 'preprocessing' in cfg else None,
        'model': build_model(cfg['model'], obj, class_num, name),
        'scorer': build_scorer(cfg['scorer'], name) if 'scorer' in cfg else None,
        'use_proba': cfg.get('use_proba', False)  # Can be absent only when regressing, and then it should be false
    }
    if 'score_merger' in cfg:
        pipeline_cfg['score_merger'] = cfg['score_merger']
    return Pipeline(**pipeline_cfg)
