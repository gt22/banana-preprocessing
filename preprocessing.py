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
Splitter = Union[ShuffleSplit, StratifiedShuffleSplit, KFold, StratifiedKFold, TimeSeriesSplit]


class Preprocessing:

    scaler: Optional[Scaler]
    splitter: Splitter

    # TODO: Cat features, encoders, resampling

    def __init__(self, scaler: ScalerType, splitter: SplitterType, kfold: int, scaler_args: dict, splitter_args: dict):
        self.scaler = self._get_scaler(scaler, scaler_args)
        self.splitter = self._get_splitter(splitter, kfold, splitter_args)

    @staticmethod
    def _get_scaler(t: ScalerType, args: dict) -> Optional[Scaler]:
        if t == ScalerType.MIN_MAX:
            return MinMaxScaler(**args)
        elif t == ScalerType.STANDARD:
            return StandardScaler(**args)
        elif t == ScalerType.NONE:
            return None
        else:
            raise ValueError(f"Unknown scaler {t}")

    @staticmethod
    def _get_splitter(t: SplitterType, n: int, args: dict) -> Splitter:
        if t == SplitterType.SHUFFLE:
            return ShuffleSplit(n, **args)
        elif t == SplitterType.STRATIFIED_SHUFFLE:
            return StratifiedShuffleSplit(n, **args)
        elif t == SplitterType.KFOLD:
            return KFold(n, **args)
        elif t == SplitterType.STRATIFIED_KFOLD:
            return StratifiedKFold(n, **args)
        elif t == SplitterType.TIME_SERIES_KFOLD:
            return TimeSeriesSplit(n, **args)
        else:
            raise ValueError(f"Unknown splitter {t}")

    def get_scaled(self, x, fit=True):
        if self.scaler is None:
            return x
        return self.scaler.fit_transform(x) if fit else self.scaler.transform(x)

    def get_split(self, x, y):
        return self.splitter.split(x, y)
