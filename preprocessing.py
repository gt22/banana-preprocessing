# %%
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit, ShuffleSplit, StratifiedShuffleSplit
from enum import Enum
from typing import Union, Optional

# %%


class ScalerType(Enum):
    NONE = 'none'
    MIN_MAX = 'minmax'
    STANDARD = 'standard'


class SplitterType(Enum):
    SHUFFLE = 'shuffle'
    STRATIFIED_SHUFFLE = 'stratifiedshuffle'
    KFOLD = 'kfold'
    STRATIFIED_KFOLD = 'stratifiedkfold'
    TIME_SERIES_KFOLD = 'timeserieskfold'


# %%
Scaler = Union[StandardScaler, MinMaxScaler]
Splitter = Union[KFold, StratifiedKFold, TimeSeriesSplit]


class Preprocessing:

    scaler: Optional[Scaler]
    splitter: Splitter

    def __init__(self, scaler: ScalerType, splitter: SplitterType, kfold: int):
        self.scaler = self._get_scaler(scaler)
        self.splitter = self._get_splitter(splitter, kfold)

    @staticmethod
    def _get_scaler(t: ScalerType) -> Optional[Scaler]:
        if t == ScalerType.MIN_MAX:
            return MinMaxScaler()
        elif t == ScalerType.STANDARD:
            return StandardScaler()
        elif t == ScalerType.NONE:
            return None
        else:
            raise ValueError(f"Unknown scaler {t}")

    @staticmethod
    def _get_splitter(t: SplitterType, n: int) -> Splitter:
        if t == SplitterType.SHUFFLE:
            return ShuffleSplit(n)
        elif t == SplitterType.STRATIFIED_SHUFFLE:
            return StratifiedShuffleSplit(n)
        elif t == SplitterType.KFOLD:
            return KFold(n)
        elif t == SplitterType.STRATIFIED_KFOLD:
            return StratifiedKFold(n)
        elif t == SplitterType.TIME_SERIES_KFOLD:
            return TimeSeriesSplit(n)
        else:
            raise ValueError(f"Unknown splitter {t}")

    def get_scaled(self, x, fit=True):
        if self.scaler is None:
            return x
        return self.scaler.fit_transform(x) if fit else self.scaler.transform(x)

    def get_split(self, x, y):
        return self.splitter.split(x, y)
