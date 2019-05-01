# %%
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit
from enum import Enum
from typing import Union
# %%


class ScalerType(Enum):
    MIN_MAX = 'minmax'
    STANDARD = 'standard'


class SplitterType(Enum):
    NORMAL = 'kfold'
    STRATIFIED = 'stratifiedkfold'
    TIME_SERIES = 'timeserieskfold'


# %%
class Preprocessing:

    scaler: Union[StandardScaler, MinMaxScaler]
    splitter: Union[KFold, StratifiedKFold, TimeSeriesSplit]

    def __init__(self, scaler: ScalerType, splitter: SplitterType, kfold: int):
        self.scaler = self._get_scaler(scaler)
        self.splitter = self._get_splitter(splitter, kfold)

    @staticmethod
    def _get_scaler(t: ScalerType):
        if t == ScalerType.MIN_MAX:
            return MinMaxScaler()
        elif t == ScalerType.STANDARD:
            return StandardScaler()
        else:
            raise ValueError(f"Unknown scaler {t}")

    @staticmethod
    def _get_splitter(t: SplitterType, n: int):
        if t == SplitterType.NORMAL:
            return KFold(n)
        elif t == SplitterType.STRATIFIED:
            return StratifiedKFold(n)
        elif t == SplitterType.TIME_SERIES:
            return TimeSeriesSplit(n)
        else:
            raise ValueError(f"Unknown splitter {t}")

    def get_scaled(self, x, fit=True):
        return self.scaler.fit_transform(x) if fit else self.scaler.transform(x)

    def get_split(self, x, y):
        return self.splitter.split(x, y)
