from umi.base_umi import UnifiedModelInterface, Objective
from typing import Optional, List, Callable
import os
from keras.models import Sequential, save_model
from keras.layers import Dense
from keras import backend as K
from keras.regularizers import Regularizer
from keras.callbacks import Callback


class MlpUmi(UnifiedModelInterface):
    model: Sequential
    callbacks: List[Callback]

    def __init__(self, objective: Objective, model_name: str,
                 layer_sizes: List[int], class_num: Optional[int] = None, activation: str = 'relu',
                 reg_fun: Callable[[int], Optional[Regularizer]] = lambda x: None,
                 compile_args: Optional[dict] = None, callbacks: Optional[List[Callback]] = None):
        super().__init__(objective, model_name, class_num)
        self.model = self._build_network(layer_sizes, objective, class_num, activation, reg_fun, compile_args or {})
        self.callbacks = callbacks or []

    @staticmethod
    def _build_network(layer_sizes: List[int], objective: Objective, class_num: Optional[int],
                       activation: str, reg_fun: Callable[[int], Optional[Regularizer]], compile_args: dict):
        if objective == Objective.CLASSIFICATION and class_num is None:
            raise ValueError("Mlp classifier requires class_num")
        nn = Sequential()
        inp = layer_sizes[0]
        if type(inp) is int:
            inp = [inp]
        nn.add(Dense(layer_sizes[1], input_shape=inp, activation=activation, kernel_regularizer=reg_fun(0)))
        for i, size in enumerate(layer_sizes[2:], 1):
            nn.add(Dense(size, activation=activation, kernel_regularizer=reg_fun(i)))
        if objective == Objective.REGRESSION:
            nn.add(Dense(1, activation='linear'))
            if 'loss' not in compile_args:
                compile_args['loss'] = 'mean_squared_error'
        elif objective == Objective.CLASSIFICATION:
            if class_num == 2:
                nn.add(Dense(1, activation='sigmoid'))
                if 'loss' not in compile_args:
                    compile_args['loss'] = 'binary_crossentropy'
            else:
                nn.add(Dense(class_num, activation='softmax'))
                if 'loss' not in compile_args:
                    compile_args['loss'] = 'categorical_crossentropy'
        if 'optimizer' not in compile_args:
            compile_args['optimizer'] = 'adam'
        nn.compile(**compile_args)
        return nn

    def fit(self, x_train, y_train, x_val, y_val, **kwargs):
        if 'callbacks' not in kwargs:
            kwargs['callbacks'] = []
        kwargs['callbacks'] += self.callbacks
        return self.model.fit(x_train, y_train, validation_data=(x_val, y_val) if x_val is not None else None, **kwargs)

    def predict(self, x, **kwargs):
        return self.model.predict(x, **kwargs)

    def predict_proba(self, x, **kwargs):
        if self.objective != Objective.CLASSIFICATION:
            raise ValueError("predict_proba for MLP models works only for classification")
        return self.model.predict_proba(x, **kwargs)

    def save(self, fold_dir, **kwargs):
        super().save(fold_dir)
        save_model(self.model, os.path.join(fold_dir, f'{self.model_name}.h5'))

    def on_train_end(self, **kwargs):
        K.clear_session()
        del self.model
        del self.callbacks
