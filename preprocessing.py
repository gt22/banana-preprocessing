# %%
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit, ShuffleSplit, StratifiedShuffleSplit
from enum import Enum
from typing import Union, Optional, List
import numpy as np
from utils import names_to_id, index_data, update_data
from pandas import DataFrame
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
    cat_features: Optional[Union[List[str], List[int]]]
    # TODO: encoders, resampling

    def __init__(self, scaler: ScalerType, splitter: SplitterType, kfold: int,
                 scaler_args: dict, splitter_args: dict, cat_features: Optional[Union[List[str], List[int]]] = None):
        self.scaler = self._get_scaler(scaler, scaler_args)
        self.splitter = self._get_splitter(splitter, kfold, splitter_args)
        self.cat_features = cat_features

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

    def _create_num_feature_index(self, x):
        if len(x.shape) <= 1:
            raise ValueError("Expected 2D array, got 1D array instead:\n"
                             f"{x}.\n"
                             "Reshape your data either using array.reshape(-1, 1) if your data has a single feature "
                             "or array.reshape(1, -1) if it contains a single sample.")
        if self.cat_features is None or len(self.cat_features) == 0:
            return x
        if isinstance(self.cat_features[0], str):
            if not isinstance(x, DataFrame):
                raise ValueError(f"Cat features are passes as names, but X is {type(x)}.\n"
                                 "Named features are only available with pandas DataFrame")
            cf = names_to_id(x, self.cat_features)
        else:
            cf = self.cat_features
        n_feat = x.shape[1]
        ind = np.ones(n_feat, np.bool)
        for f in cf:
            ind[f] = False
        return ind

    def get_scaled(self, x, fit=True, copy=True):
        if copy:
            x = x.copy()
        if self.scaler is None:
            return x
        num_ind = self._create_num_feature_index(x)
        x_num = index_data(x, num_ind)
        if fit:
            self.scaler.fit(x_num)
        x_num = self.scaler.transform(x_num, copy=False)
        update_data(x, num_ind, x_num)
        return x

    def get_split(self, x, y):
        return self.splitter.split(x, y)
