from sklearn.base import is_classifier, clone
from sklearn.utils import check_scalar, check_random_state
from sklearn.model_selection import train_test_split

from .base import BaseLearner
from ..base import RegressorMixin, ClassifierMixin


class XLearner(BaseLearner):
    def __init__(self,
                 *,
                 estimator=None,
                 estimator_alpha_t=None,
                 estimator_alpha_c=None,
                 estimator_beta_t=None,
                 estimator_beta_c=None,
                 propencity_score: float=None,
                 propencity_estimator: float=None,
                 random_state: int = None):
        super().__init__(estimator=estimator,
                         estimators_params=('estimator_alpha_t',
                                            'estimator_alpha_c',
                                            'estimator_beta_t',
                                            'estimator_beta_c'),
                         propencity=True,
                         propencity_score=propencity_score,
                         propencity_estimator=propencity_estimator,
                         random_state=random_state)
        self.estimator = estimator
        self.estimator_alpha_t = estimator_alpha_t
        self.estimator_alpha_c = estimator_alpha_c
        self.estimator_beta_t = estimator_beta_t
        self.estimator_beta_c = estimator_beta_c
        self.propencity_score = propencity_score
        self.propencity_estimator = propencity_estimator
        self.random_state = random_state

    def _check_params(self):
        params = super()._check_params()

        params['random_state'] = check_random_state(self.random_state)

        return params

    def _fit_group(self, group, X, y, w,
                   random_state,
                   fit_params):
        estimator_alpha_t = clone(self.estimator_alpha_t)
        estimator_alpha_c = clone(self.estimator_alpha_c)
        estimator_beta_t = clone(self.estimator_beta_t)
        estimator_beta_c = clone(self.estimator_beta_c)

        estimator_alpha_t.fit(X[w == 1], y[w == 1],
                              **fit_params.get('estimator_alpha_t', {}))
        estimator_alpha_c.fit(X[w == 0], y[w == 0],
                              **fit_params.get('estimator_alpha_c', {}))

        if is_classifier(self):
            dt = y[w == 1] - estimator_alpha_c.predict_proba(X[w == 1])[:, 1]
            dc = estimator_alpha_t.predict_proba(X[w == 0])[:, 1] - y[w == 0]
        else:
            dt = y[w == 1] - estimator_alpha_c.predict(X[w == 1])
            dc = estimator_alpha_t.predict(X[w == 0]) - y[w == 0]

        estimator_beta_t.fit(X[w == 1], dt)
        estimator_beta_c.fit(X[w == 0], dc)

        self.estimators[group - 1] = (estimator_beta_t, estimator_beta_c,
                                      self.estimators[group - 1])

    def _predict_group(self, group, X, p_score, **kwargs):
        estimator_t = self.estimators[group - 1][0]
        estimator_c = self.estimators[group - 1][1]

        if is_classifier(self):
            pred_t = estimator_t.predict_proba(X)[:, 1]
            pred_c = estimator_c.predict_proba(X)[:, 1]
        else:
            pred_t = estimator_t.predict(X)
            pred_c = estimator_c.predict(X)

        return p_score * pred_c + (1 - p_score) * pred_t  


class XClassifier(XLearner, ClassifierMixin):
    pass


class XRegressor(XLearner, RegressorMixin):
    pass
