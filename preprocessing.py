# %%
from enum import Enum
from typing import Union, Optional, List

import numpy as np
from pandas import DataFrame
from scipy.sparse.base import issparse
from scipy.sparse.csr import csr_matrix
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit, ShuffleSplit, StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from utils import names_to_id, index_data, concatenate


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


class EncoderType(Enum):
    NONE = 'none'
    ONE_HOT = 'onehot'


# %%
Scaler = Union[StandardScaler, MinMaxScaler]
Splitter = Union[ShuffleSplit, StratifiedShuffleSplit, KFold, StratifiedKFold, TimeSeriesSplit]
Encoder = Union[OneHotEncoder]


class Preprocessing:

    scaler: Optional[Scaler]
    splitter: Splitter
    encoder: Encoder
    cat_features: Optional[Union[List[str], List[int]]]
    # TODO: resampling

    def __init__(self, scaler: ScalerType, splitter: SplitterType, kfold: int, encoder: EncoderType,
                 scaler_args: dict, splitter_args: dict, encoder_args: dict,
                 cat_features: Optional[Union[List[str], List[int]]] = None):
        self.scaler = self._get_scaler(scaler, scaler_args)
        self.splitter = self._get_splitter(splitter, kfold, splitter_args)
        self.encoder = self._get_encoder(encoder, encoder_args)
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

    @staticmethod
    def _get_encoder(t: EncoderType, args: dict) -> Optional[Encoder]:
        if t == EncoderType.NONE:
            return None
        elif t == EncoderType.ONE_HOT:
            return OneHotEncoder(**args)
        else:
            raise ValueError(f"Unknown encoder {t}")

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

    def _encode(self, x_cat, fit: bool, sparse: Optional[bool]):
        if self.encoder is None:
            return x_cat
        if isinstance(self.encoder, OneHotEncoder):
            if fit:
                self.encoder.fit(x_cat)
            x_tran = self.encoder.transform(x_cat)
            if sparse is True or (sparse is None and issparse(x_cat)):
                return x_tran
            if isinstance(x_cat, DataFrame):
                return DataFrame(x_tran.toarray(), columns=self.encoder.get_feature_names(x_cat.columns))
            else:
                return x_tran.toarray()

    def _scale(self, x_num, fit: bool, sparse):
        if self.scaler is None:
            return x_num
        if fit:
            self.scaler.fit(x_num)
        x_tran = self.scaler.transform(x_num, copy=False)
        if sparse is True:
            if issparse(x_num):
                return type(x_num)(x_tran)
            else:
                return csr_matrix(x_tran)
        elif sparse is None and issparse(x_num):
            return type(x_num)(x_tran)
        if isinstance(x_num, DataFrame):
            return DataFrame(x_tran, columns=x_num.columns)
        else:
            return x_tran

    def preproc(self, x, fit: bool=True, copy: bool=True, sparse: Optional[bool] = None):
        if copy:
            x = x.copy()

        num_ind = self._create_num_feature_index(x)

        x_num = self._scale(index_data(x, num_ind), fit, sparse)
        x_cat = self._encode(index_data(x, ~num_ind), fit, sparse)

        return concatenate(x_num, x_cat, axis=1)
    
    def get_split(self, x, y):
        return self.splitter.split(x, y)
