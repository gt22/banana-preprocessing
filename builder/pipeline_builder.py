from typing import Optional, Union, List

from builder.umi_builder import builder_map
from pipeline import Pipeline
from preprocessing import Preprocessing, SplitterType, ScalerType, EncoderType
from scorer.scorer import Scorer, SaveTactics
from umi.base_umi import Objective, UnifiedModelInterface

# TODO: Gird/RandomSearch, hyper/bayesopt
# TODO: Dim reduction

config_example = {
    'objective': 'classification',
    'name': 'cb_model',
    'cat_feaures': ['a', 's', 'd'],
    'preprocessing': {
        'scaler': 'minmax',
        'splitter': 'shuffle',
        'kfold': 3,
        'splitter_args': {
            'test_size': 0.3
        }
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


def build_preprocessing(cfg: dict, cat_features: Optional[Union[List[str], List[int]]]) -> Preprocessing:
    scaler = ScalerType(cfg.get('scaler', 'none'))
    splitter = SplitterType(cfg.get('splitter', 'kfold'))
    encoder = EncoderType(cfg.get('encoder', 'none'))
    kfold = cfg.get('kfold', DEFAULT_KFOLD)
    scaler_args = cfg.get('scaler_args', {})
    splitter_args = cfg.get('splitter_args', {})
    encoder_args = cfg.get('encoder_args', {})
    return Preprocessing(scaler, splitter, kfold, encoder, scaler_args, splitter_args, encoder_args, cat_features)


def build_model(cfg: dict, objective: Objective, class_num: Optional[int], name: str,
                cat_features: Optional[Union[List[str], List[int]]]) -> UnifiedModelInterface:
    model_type = cfg['type']
    if model_type not in builder_map:
        raise ValueError(f"Unknown type '{model_type}'")
    model_cfg = cfg.copy()
    model_cfg.pop('type')
    model_cfg.pop('name', None)
    return builder_map[model_type](model_cfg, objective, name, class_num, cat_features)


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
    cat_features = cfg.get('cat_features', None)
    pipeline_cfg = {
        'preproc': build_preprocessing(cfg['preprocessing'], cat_features) if 'preprocessing' in cfg else None,
        'model': build_model(cfg['model'], obj, class_num, name, cat_features),
        'scorer': build_scorer(cfg['scorer'], name) if 'scorer' in cfg else None,
        'use_proba': cfg.get('use_proba', False)  # Can be absent only when regressing, and then it should be false
    }
    if 'score_merger' in cfg:
        pipeline_cfg['score_merger'] = cfg['score_merger']
    return Pipeline(**pipeline_cfg)
