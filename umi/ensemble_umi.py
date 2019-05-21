from umi import UnifiedModelInterface, Objective
from typing import Optional, Union, List, Callable, Any, Dict
from multiprocessing import Pool
import numpy as np
import os
from functools import partial

ModelMerger = Callable[[List[Any]], Any]

known_mergers: Dict[str, ModelMerger] = {
    'mean': partial(np.mean, axis=0),
    'median': partial(np.median, axis=0)
}


class EnsembleUMI(UnifiedModelInterface):

    models: List[UnifiedModelInterface]
    merger: ModelMerger

    def __init__(self, models: List[UnifiedModelInterface], objective: Objective, model_name: str,
                 merger: Union[ModelMerger, str] = 'mean',
                 class_num: Optional[int] = None, cat_features: Optional[Union[List[str], List[int]]] = None, **kwargs):
        super().__init__(objective, model_name, class_num, cat_features)
        self.models = models
        self.merger = known_mergers[merger] if type(merger) == str else merger

    def fit(self, x_train, y_train, x_val, y_val, **kwargs):
        n_jobs = kwargs.pop('ensemble_n_jobs', 1)
        if n_jobs == 1:
            self._fit_sync(x_train, y_train, x_val, y_val, **kwargs)
        else:
            self._fit_async(x_train, y_train, x_val, y_val, n_jobs, **kwargs)

    def _fit_sync(self, x_train, y_train, x_val, y_val, **kwargs):
        for m in self.models:
            m.fit(x_train, y_train, x_val, y_val, **kwargs)

    def _fit_async(self, x_train, y_train, x_val, y_val, n_jobs, **kwargs):
        with Pool(n_jobs) as pool:
            # Fitting models in different processes doesn't fit them in main process, so they must be returned and saved
            self.models = pool.map(partial(self._fit_model, x_train, y_train, x_val, y_val, **kwargs), self.models)

    @staticmethod
    def _fit_model(x_train, y_train, x_val, y_val, m, **kwargs):
        m.fit(x_train, y_train, x_val, y_val, **kwargs)
        return m

    def predict(self, x, **kwargs):
        n_jobs = kwargs.pop('ensemble_n_jobs', 1)
        if n_jobs == 1:
            pred = self._predict_sync(x, **kwargs)
        else:
            pred = self._predict_async(x, **kwargs)
        return self.merger(pred)

    def _predict_sync(self, x, **kwargs):
        return [m.predict(x, **kwargs) for m in self.models]

    def _predict_async(self, x, n_jobs, **kwargs):
        with Pool(n_jobs) as pool:
            ret = pool.map(partial(self._predict_model, x, **kwargs), self.models)
        return ret

    @staticmethod
    def _predict_model(x, m, **kwargs):
        return m.predict(x, **kwargs)

    def predict_proba(self, x, **kwargs):
        n_jobs = kwargs.pop('ensemble_n_jobs', 1)
        if n_jobs == 1:
            pred = self._predict_proba_sync(x, **kwargs)
        else:
            pred = self._predict_proba_async(x, **kwargs)
        return self.merger(pred)

    def _predict_proba_sync(self, x, **kwargs):
        return [m.predict_proba(x, **kwargs) for m in self.models]

    def _predict_proba_async(self, x, n_jobs, **kwargs):
        with Pool(n_jobs) as pool:
            ret = pool.map(partial(self._predict_proba_model, x, **kwargs), self.models)
        return ret

    @staticmethod
    def _predict_proba_model(x, m, **kwargs):
        return m.predict_proba(x, **kwargs)

    def save(self, fold_dir, **kwargs):
        ensemble_dir = os.path.join(fold_dir, self.model_name)
        if not os.path.exists(ensemble_dir):
            os.makedirs(ensemble_dir)
        for m in self.models:
            m.save(ensemble_dir, **kwargs)

    def on_train_end(self, **kwargs):
        for m in self.models:
            m.on_train_end(**kwargs)
        del self.models
        del self.merger
