from .hyperoptimizer import Hyperoptimizer
from sklearn.model_selection import ParameterGrid
from multiprocessing import Pool
from builder import build_pipeline
from scorer.scorer import ScoreData, CleanScoreData, ImprovementCriterion
from typing import Tuple


class RandomSearchOptimizer(Hyperoptimizer):

    def __init__(self, pipeline_conf, param_space: dict, criterion: ImprovementCriterion):
        super().__init__(pipeline_conf)
        self.searcher = ParameterGrid(param_space)
        self.criterion = criterion

    def _check_params(self, x, y, params: dict) -> Tuple[dict, ScoreData]:
        cfg = self.get_parametrized_conf(params)
        pipe = build_pipeline(cfg)
        score: CleanScoreData = pipe.run(x, y)

        return params, pipe.scorer.restore(score)

    def start_search(self, x, y, search_jobs: int = 1) -> Tuple[dict, ScoreData]:
        if search_jobs == 1:
            best_score = None
            best_params = None
            for params in self.searcher:
                _, score = self._check_params(x, y, params)
                if best_score is None or self.criterion(score, best_score):
                    best_score = score
                    best_params = params
            return best_params, best_score
        else:
            pool = Pool(search_jobs)
            score_data = pool.map(lambda p: self._check_params(x, y, p), self.searcher)
            best_score = None
            best_params = None
            for s, p in score_data:
                if best_score is None or self.criterion(s, best_score):
                    best_score = s
                    best_params = p
            return best_score, best_params
