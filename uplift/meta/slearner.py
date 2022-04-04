import numpy as np

from sklearn.base import is_classifier, clone

from .base import BaseLearner
from ..base import RegressorMixin, ClassifierMixin


class SLearner(BaseLearner):
    def __init__(self,
                 *,
                 estimator=None,):
        super().__init__(estimator=estimator,
                         estimators_params=tuple())
        self.esimator = estimator

    def _check_params(self):
        return super()._check_params()

    def _fit_group(self, group, X, y, w, fit_params):
        Xw = np.hstack((X, w.reshape(-1, 1)))

        estimator = clone(self.estimator)
        estimator.fit(Xw, y, **fit_params)

        self.estimators[group - 1] = estimator

    def _predict_group(self, group, X, **kwargs):
        n_samples, _ = X.shape

        w_shape = (n_samples, 1)
        Xt = np.hstack((X, np.ones(w_shape)))
        Xc = np.hstack((X, np.zeros(w_shape)))

        estimator = self.estimators[group - 1]
        if is_classifier(self):
            pred_t = estimator.predict_proba(Xt)[:, 1]
            pred_c = estimator.predict_proba(Xc)[:, 1]
        else:
            pred_t = estimator.predict(Xt)
            pred_c = estimator.predict(Xc)
        
        return pred_t - pred_c


class SClassifier(SLearner, ClassifierMixin):
    pass


class SRegressor(SLearner, RegressorMixin):
    pass
