# %%
import pandas as pd
import numpy as np
from builder import build_pipeline
from optimizer.randomsearch import RandomSearchOptimizer
from scipy.stats.distributions import randint
from scorer.scorer import any_improve_criterion
# %#%
df = pd.read_csv('data/titanic/train.csv', index_col='PassengerId')
# %#%
df.drop(['Cabin', 'Ticket', 'Name'], axis=1, inplace=True)
# %#%
df['Age'].fillna(round(df['Age'].mean()), inplace=True)
df['Embarked'].fillna(df['Embarked'].value_counts().idxmax(), inplace=True)
# %#%
cat_features = ['Pclass', 'Sex', 'Embarked']
# %%
cb_cfg = {
    'type': 'catboost',
    'iterations': '##iter_count',
    'eval_metric': 'Accuracy',
    'use_best_model': True,
    'verbose': True,
    'random_seed': 6741
}

pipeline_cfg = {
    'objective': 'classification',
    'cat_features': cat_features,
    'name': 'titanic',
    'preprocessing': {
        'scaler': 'standard',
        'encoder': 'none',
        'splitter': 'shuffle',
        'kfold': 1
    },
    'model': cb_cfg,
    'scorer': {
        'metrics': 'accuracy',
        'save': 'iob'
    },
    'use_proba': False
}
# %#%
X = df.drop('Survived', axis=1)
y = df['Survived']
# %#%
param_space = {
    'iter_count': randint(10, 1000)
}
# %%
optimizer = RandomSearchOptimizer(pipeline_cfg, param_space, 5, any_improve_criterion())
# %#%
best_params, best_score = optimizer.start_search(X, y)
# %%
best_pipe = build_pipeline(optimizer.get_parametrized_conf(best_params))
# %%
score = best_pipe.run(X, y)
print(next(iter(score.values())))
# %%
