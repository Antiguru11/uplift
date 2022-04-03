from sklearn.base import is_classifier, clone

from .base import BaseLearner
from ..base import RegressorMixin, ClassifierMixin


class TLearner(BaseLearner):
    def __init__(self,
                 *,
                 estimator=None,
                 estimator_t=None,
                 estimator_c=None):
        super().__init__(estimator=estimator,
                         estimators_params=('estimator_t',
                                            'estimator_c',))
        self.estimator = estimator
        self.estimator_t = estimator_t
        self.estimator_c = estimator_c

    def _check_params(self):
        return super()._check_params()

    def _fit_group(self, group, X, y, w, fit_params):
        estimator_t = clone(self.estimator_t)
        estimator_c = clone(self.estimator_c)

        estimator_t.fit(X[w == 1], y[w == 1],
                        **fit_params.get('estimator_t', {}))
        estimator_c.fit(X[w == 0], y[w == 0],
                        **fit_params.get('estimator_c', {}))

        self.estimators[group - 1] = (estimator_t, estimator_c)

    def _predict_group(self, group, X, **kwargs):
        estimator_t = self.estimators[group - 1][0]
        estimator_c = self.estimators[group - 1][1]

        if is_classifier(self):
            pred_t = estimator_t.predict_proba(X)[:, 1]
            pred_c = estimator_c.predict_proba(X)[:, 1]
        else:
            pred_t = estimator_t.predict(X)
            pred_c = estimator_c.predict(X)

        return pred_t - pred_c


class TClassifier(TLearner, ClassifierMixin):
    pass


class TRegressor(TLearner, RegressorMixin):
    pass
