from sklearn.base import is_classifier, clone

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
        return super()._check_params()

    def _fit_group(self, group, X, y, w, fit_params):
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

        self.estimators[group - 1] = (estimator_beta_t, estimator_beta_c)

    def _predict_group(self, group, X, **kwargs):
        estimator_t = self.estimators[group - 1][0]
        estimator_c = self.estimators[group - 1][1]

        pred_t = estimator_t.predict(X)
        pred_c = estimator_c.predict(X)

        p_score = self._predict_propencity(group, X)

        return p_score * pred_c + (1 - p_score) * pred_t  


class XClassifier(XLearner, ClassifierMixin):
    def _make_estimators(self):
        if self.estimator is not None:
            for e in self.estimators_params:
                if 'alpha' in e:
                    setattr(self, e, clone(self.estimator[0]))
                if 'beta' in e:
                    setattr(self, e, clone(self.estimator[1]))

        if len(self.estimators_params) != 0:
            self.estimator = None
        else:
            self.estimators_params = ('estimator',)

    def _check_params(self):
        params = dict()

        for e in self.estimators_params:
            if getattr(self, e) is None:
                raise ValueError(f'Estimator {e} is None')

            if 'alpha' in e:
                if is_classifier(self) != is_classifier(getattr(self, e)):
                    raise ValueError(f'Estimator {e} must be '
                                    + ('classifier' if is_classifier(self) else 'regressor'))
            if 'beta' in e:
                if is_classifier(getattr(self, e)):
                    raise ValueError(f'Estimator {e} must be regressor')

        if self.propencity:
            if self.propencity_score is None:
                if self.propencity_estimator is None:
                    raise ValueError('Estimator for propencity is None')
                if not is_classifier(self.propencity_estimator):
                    raise ValueError('Estimator for propencity must be classifier')
            else:
                if not isinstance(self.propencity_score, list):
                    self.propencity_score = [self.propencity_score]
                
                if len(self.propencity_score) != self.n_groups:
                    raise ValueError('Propencity vector must have same lenght as groups')
                for score in self.propencity_score:
                    check_scalar(score,
                                 'propencity_score', float,
                                 min_val=0, max_val=1,
                                 include_boundaries='neither')

        return params


class XRegressor(XLearner, RegressorMixin):
    pass
