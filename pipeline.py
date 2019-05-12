from umi import UnifiedModelInterface
from preprocessing import Preprocessing
from scorer.scorer import Scorer, CleanScoreData
from typing import Optional, Union
from scorer.score_mergers import known_mergers, ScoreMerger


class Pipeline:

    preproc: Optional[Preprocessing]
    model: UnifiedModelInterface
    scorer: Optional[Scorer]
    score_merger: ScoreMerger

    def __init__(self, preproc: Optional[Preprocessing], model: UnifiedModelInterface, use_proba: bool,
                 scorer: Optional[Scorer], score_merger: Union[str, ScoreMerger] = 'mean'):
        self.preproc = preproc
        self.model = model
        self.use_proba = use_proba
        self.scorer = scorer
        self.score_merger = known_mergers[score_merger] if type(score_merger) is str else score_merger

    def run(self, x, y) -> Optional[CleanScoreData]:
        if self.preproc is not None:
            x = self.preproc.get_scaled(x)
            scores = []
            for train_id, val_id in self.preproc.get_split(x, y):
                self.fit(x[train_id], y[train_id], x[val_id], y[val_id])
                val_pred = self.predict(x[val_id])
                if self.scorer is not None:
                    scores.append(self.scorer.score(y[val_id], val_pred, record_score=False))
            merged_score = self.score_merger(scores)
            if self.scorer is not None:
                self.scorer.record(self.scorer.restore(merged_score))
            return merged_score
        else:
            self.fit(x, y, None, None)
            return self.scorer.score(y, self.predict(x))

    def fit(self, x_train, y_train, x_val, y_val):
        self.model.fit(x_train, y_train, x_val, y_val)

    def predict(self, x):
        return self.model.predict_proba(x) if self.use_proba else self.model.predict(x)