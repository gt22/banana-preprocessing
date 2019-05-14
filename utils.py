from pandas import DataFrame, Series
from typing import List
import numpy as np


def names_to_id(data: DataFrame, names: List[str]) -> List[int]:
    c = data.columns.tolist()
    return [c.index(x) for x in names]


def index_data(x, ind):
    if isinstance(x, DataFrame):
        return x.loc[:, ind]
    else:
        return x[:, ind]


def index_entries(x, ind, reset_index=False, copy=True):
    if copy:
        x = x.copy()
    ind = np.array(ind)
    if isinstance(x, DataFrame):
        if reset_index:
            index_name = x.index.name
            x = x.reset_index()
            x = x.loc[ind, :]
            x = x.set_index(index_name)
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
