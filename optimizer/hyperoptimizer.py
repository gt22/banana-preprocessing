from abc import ABC, abstractmethod
from copy import deepcopy
from scorer.scorer import ScoreData
from typing import Tuple


class Hyperoptimizer(ABC):

    pipeline_conf: dict

    def __init__(self, pipeline_conf: dict):
        self.pipeline_conf = pipeline_conf

    @abstractmethod
    def start_search(self, x, y, search_jobs: int = 1) -> Tuple[dict, ScoreData]:
        pass

    def get_parametrized_conf(self, params: dict) -> dict:
        param_conf = deepcopy(self.pipeline_conf)
        to_process = [param_conf]
        while to_process:
            cur = to_process.pop(0)
            for k, v in cur.items() if isinstance(cur, dict) else enumerate(cur):
                if isinstance(v, str):
                    if v.startswith('##'):
                        cur[k] = params[v[2:]]
                elif isinstance(v, dict) or isinstance(v, list):
                    to_process.append(v)
        return param_conf
