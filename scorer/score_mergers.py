from collections import defaultdict
from typing import List, Callable, Dict

import numpy as np

from scorer.scorer import CleanScoreData

ScoreMerger = Callable[[List[CleanScoreData]], CleanScoreData]


def mean_merger(scores: List[CleanScoreData]) -> CleanScoreData:
    tmp = defaultdict(list)
    for sc in scores:
        for k, v in sc.items():
            tmp[k].append(v)
    ret = {}
    for k in tmp:
        ret[k] = np.mean(tmp[k])
    # noinspection PyTypeChecker
    return ret


def mean_std_merger(scores: List[CleanScoreData]) -> CleanScoreData:
    tmp = defaultdict(list)
    for sc in scores:
        for k, v in sc.items():
            tmp[k] += v
    ret = {}
    for k in tmp:
        ret[f'{k}_mean'] = np.mean(tmp[k])
        ret[f'{k}_std'] = np.std(tmp[k])
    # noinspection PyTypeChecker
    return ret


def min_merger(scores: List[CleanScoreData]) -> CleanScoreData:
    ret = defaultdict(lambda: float('inf'))
    for sc in scores:
        for k, v in sc.items():
            if v < ret[k]:
                ret[k] = v
    return ret


def max_merger(scores: List[CleanScoreData]) -> CleanScoreData:
    ret = defaultdict(lambda: float('-inf'))
    for sc in scores:
        for k, v in sc.items():
            if v > ret[k]:
                ret[k] = v
    return ret


known_mergers: Dict[str, ScoreMerger] = {
    'mean': mean_merger,
    'mean+std': mean_std_merger,
    'max': max_merger,
    'min': min_merger
}
