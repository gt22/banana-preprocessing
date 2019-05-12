from typing import Callable, Union, List, Any, Dict, Tuple
from enum import Enum
from math import sqrt
import os
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, \
    mean_absolute_error, mean_squared_error, mean_squared_log_error, r2_score, recall_score, precision_score


class MetricType(Enum):
    HIGHER_IS_BETTER = '+'
    LOWER_IS_BETTER = '-'
    HIGH = '+'
    LOW = '-'


MetricFunction = Tuple[Callable[[Any, Any], float], MetricType]
NamedMetric = Tuple[str, MetricFunction]
ScoreData = Dict[NamedMetric, float]
CleanScoreData = Dict[str, float]
ImprovementCriterion = Callable[[ScoreData, ScoreData], bool]

H = MetricType.HIGH
L = MetricType.LOW
known_metrics: Dict[str, MetricFunction] = {
    'accuracy': (accuracy_score, H),
    'roc_auc': (roc_auc_score, H),
    'precision': (precision_score, H),
    'recall': (recall_score, H),
    'f1': (f1_score, H),
    'mae': (mean_absolute_error, L),
    'mse': (mean_squared_error, L),
    'rmse': (lambda y_true, y_pred: sqrt(mean_squared_error(y_true, y_pred)), L),
    'msle': (mean_squared_log_error, L),
    'rmsle': (lambda y_true, y_pred: sqrt(mean_squared_log_error(y_true, y_pred)), L),
    'r2': (r2_score, H)
}
del H, L


def compare_scores(a: float, b: float, m: MetricType, eps: float) -> int:
    # a == b
    if abs(a - b) < eps:
        return 0

    # a != b
    if m == MetricType.HIGH:
        # 1 if a > b else -1
        return 1 if a - b > 0 else -1
    else:
        # 1 if a < b else -1
        return 1 if a - b < 0 else -1


def any_improve_criterion(eps=1e-5) -> ImprovementCriterion:
    def criterion(a, b) -> bool:
        for n in a:
            t: MetricType = n[1][1]
            if compare_scores(a[n], b[n], t, eps) == 1:
                return True
        return False

    return criterion


def all_improve_criterion(eps=1e-5) -> ImprovementCriterion:
    def criterion(a: ScoreData, b: ScoreData) -> bool:
        for n in a:
            t: MetricType = n[1][1]
            if compare_scores(a[n], b[n], t, eps) != 1:
                return False
        return True

    return criterion


known_criterions: Dict[str, ImprovementCriterion] = {
    'any_improve': any_improve_criterion(),
    'all_improve': all_improve_criterion(),
}


class SaveTactics(Enum):
    NONE = 'none'
    IMPROVE_OVER_BEST = 'iob'
    IMPROVE_OVER_PREV = 'iop'
    ALL = 'all'


class Scorer:
    name: str
    metrics: Dict[str, NamedMetric]
    criterion: ImprovementCriterion
    save_tactics: SaveTactics
    save_file: str

    history: List[ScoreData] = []
    best_score: ScoreData = None

    def __init__(self, name: str, metrics: List[Union[str, NamedMetric]],
                 improvement_criterion: Union[str, ImprovementCriterion] = 'any_improve',
                 save_tactics: SaveTactics = SaveTactics.NONE, save_dir: str = 'score'):
        metrics_dict = {}
        for i, e in enumerate(metrics):
            if type(e) is str:
                if e not in known_metrics:
                    raise NotImplementedError(f"Unknown metric '{e}'.")
                metrics_dict[e] = (e, known_metrics[e])
            else:
                metrics_dict[e[0]] = e
        if type(improvement_criterion) is str:
            if improvement_criterion not in known_criterions:
                raise NotImplementedError(f"Unknown criterion '{improvement_criterion}'")
            improvement_criterion = known_criterions[improvement_criterion]
        self.name = name
        self.metrics = metrics_dict
        self.criterion = improvement_criterion
        self.save_tactics = save_tactics
        self.save_file = os.path.join(save_dir, f'{name}_score.csv')
        self._init_file(save_dir)

    def _init_file(self, save_dir: str):
        if os.path.exists(self.save_file):
            self._load_history()
        else:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            header = ','.join(n for n in self.metrics.keys()) + '\n'
            with open(self.save_file, 'w') as f:
                f.write(header)

    def _load_history(self):
        history = []
        best_score = None
        with open(self.save_file) as f:
            name = f.readline()[:-1].split(',')

            if len(name) != len(self.metrics):
                raise ValueError(f"History file {self.save_file} doesn't match current metrics!")
            for n in name:
                if n not in self.metrics:
                    raise ValueError(f"History file {self.save_file} doesn't match current metrics!")
            for line in f.readlines():
                score: ScoreData = dict(zip(self.metrics.values(), [float(x) for x in line.split(',')]))
                history.append(score)
                if best_score is None or self.criterion(score, best_score):
                    best_score = score
        self.history = history
        self.best_score = best_score

    def save(self, score: ScoreData):
        line = ','.join(str(v) for v in score.values()) + '\n'
        with open(self.save_file, 'a') as f:
            f.write(line)

    def should_save(self, score: ScoreData) -> bool:
        if self.save_tactics == SaveTactics.ALL:
            return True
        elif self.save_tactics == SaveTactics.IMPROVE_OVER_PREV:
            if len(self.history) == 0:
                return True
            return self.criterion(score, self.history[-1])
        elif self.save_tactics == SaveTactics.IMPROVE_OVER_BEST:
            if self.best_score is None:
                return True
            return self.criterion(score, self.best_score)
        else:
            return False

    def record(self, s: ScoreData):
        if self.should_save(s):
            self.save(s)
        if self.best_score is None or self.criterion(s, self.best_score):
            self.best_score = s
        self.history.append(s)

    def restore(self, named_scores: CleanScoreData) -> ScoreData:
        return {self.metrics[n]: s for n, s in named_scores.items()}

    def score(self, y_true, y_pred, clean: bool = True, record_score: bool = True) -> Union[ScoreData, CleanScoreData]:
        s = {m: m[1][0](y_true, y_pred) for m in self.metrics.values()}
        if record_score:
            self.record(s)
        return {k[0]: v for k, v in s.items()} if clean else s
