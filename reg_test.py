# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from builder import build_pipeline
# %#%
df = pd.read_csv("data/house-prices-advanced-regression-techniques/train.csv", index_col='Id')
test_df = pd.read_csv("data/house-prices-advanced-regression-techniques/test.csv", index_col='Id')
# %#%
nans = ((df.isnull().sum() + test_df.isnull().sum()) / (df.shape[0] + test_df.shape[0])).sort_values(ascending=False)
to_drop = nans[nans > 0.5].index
fills = {
    'LotFrontage': "#mean",
    "GarageYrBlt": 0,
    "MasVnrArea": 0
}
# %#%
df.drop(to_drop, inplace=True, axis=1)
test_df.drop(to_drop, inplace=True, axis=1)
for n in nans[nans > 0].index:
    if n in to_drop:
        continue
    f = fills.get(n, "None" if df[n].dtype == object else 0)
    if f == '#mean':
        f = df[n].mean()
    df[n].fillna(f, inplace=True)
    test_df[n].fillna(f, inplace=True)
# %#%
cat_features = df.select_dtypes(include=['object']).columns.tolist() + \
                ['MSSubClass', 'OverallQual', 'OverallCond']

encoder = LabelEncoder()
for c in cat_features:
    encoder.fit(pd.concat([df[c], test_df[c]]))
    df[c] = encoder.transform(df[c])
    test_df[c] = encoder.transform(test_df[c])
# %#%
df['SalePrice'] = np.log(df['SalePrice'])
# %#%


def names_to_id(n, d):
    c = d.columns.tolist()
    return [c.index(x) for x in n]


# %#%
pipeline_cfg = {
    'objective': 'regression',
    'cat_features': cat_features,
    'preprocessing': {
        'scaler': 'standard',
        'splitter': 'kfold',
        'kfold': 3
    },
    'model': {
        'type': 'catboost',
        'iterations': 1000,
        'eval_metric': 'RMSE',
        'random_seed': 6741,
        'use_best_model': True,
        'verbose': True
    },
    'scorer': {
        'metrics': 'rmse',
        'save': 'iob'
    }
}
# %#%
pipe = build_pipeline(pipeline_cfg)
# %%
score = pipe.run(df.drop('SalePrice', axis=1), df['SalePrice'])
print(f"Cur: {score['rmse']}")
if len(pipe.scorer.history) > 1:
    m = pipe.scorer.metrics['rmse']
    prev = pipe.scorer.history[-2][m]
    print(f"Prev: {prev}")
    print(f"Diff from prev: {score['rmse'] - prev}")
    best = pipe.scorer.best_score[m]
    print(f'Best: {best}')
    print(f'Diff from best: {score["rmse"] - best}')
