from typing import List

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from scipy.sparse import hstack, vstack
from scipy.sparse.base import issparse


def names_to_id(data: DataFrame, names: List[str]) -> List[int]:
    c = data.columns.tolist()
    return [c.index(x) for x in names]


def index_data(x, ind):
    if isinstance(x, DataFrame):
        return x.loc[:, ind]
    else:
        return x[:, ind]


def concatenate(a, b, axis=0, prefer_sparse=True):
    if isinstance(a, np.ndarray):
        if isinstance(b, np.ndarray):
            return np.concatenate((a, b), axis=axis)
        if isinstance(b, DataFrame):
            return np.concatenate((a, b.values), axis=axis)
        if issparse(b):
            if prefer_sparse:
                return (vstack if axis == 0 else hstack)((type(b)(a), b))
            else:
                return np.concatenate((a, b.toarray().reshape(a.shape)), axis=axis)
    elif isinstance(a, DataFrame):
        if isinstance(b, DataFrame):
            return pd.concat((a, b), axis=axis)
        else:
            return concatenate(a.values, b, axis, prefer_sparse)
    elif issparse(a):
        if issparse(b):
            return (vstack if axis == 0 else hstack)((a, b))
        if isinstance(b, DataFrame):
            b = b.values
        return (vstack if axis == 0 else hstack)((a, type(a)(b)))
    else:
        raise ValueError(f"Unknown object {a}")


def index_entries(x, ind, reset_index=False, copy=True):
    if copy:
        x = x.copy()
    ind = np.array(ind)
    if isinstance(x, DataFrame):
        if reset_index:
            index_name = x.index.name
            x = x.reset_index()
            x = x.loc[ind, :]
            x = x.set_index(index_name if index_name is not None else 'index')
            return x
        else:
            return x.loc[ind, :]
    elif isinstance(x, Series):
        if reset_index:
            index_name = x.index.name
            x = x.reset_index()
            x = x.loc[ind, :]  # Series is converted to DataFrame after index reset, so loc is 2D
            x = x.set_index(index_name)
            return x
        else:
            return x.loc[ind]
    else:
        return x[ind, :]


def update_data(x, ind, y):
    if isinstance(x, DataFrame):
        x.loc[:, ind] = y
    else:
        x[:, ind] = y
