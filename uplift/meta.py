from abc import ABCMeta

import numpy as np
from sklearn.base import BaseEstimator, MetaEstimatorMixin, is_classifier, clone
from sklearn.utils import check_X_y, check_scalar, check_random_state
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted

from .base import UpliftMixin, ClassifierMixin, RegressorMixin


class _SLearner(BaseEstimator, MetaEstimatorMixin, UpliftMixin, metaclass=ABCMeta):
    def __init__(self,
                 estimator,):
        self.estimator = estimator

    def fit(self, X, y, w, **fit_params):
        X, y = self._validate_data(X, y, force_all_finite='allow-nan')
        _, w = check_X_y(X, w, force_all_finite='allow-nan')

        if is_classifier(self) != is_classifier(self.estimator):
            raise ValueError('Estimator must be same type')

        Xw = np.hstack((X, w.reshape(-1, 1)))

        self.estimator.fit(Xw, y, **fit_params)

        return self

    def predict(self, X, **kwargs):
        check_is_fitted(self)
        X = self._validate_data(X, reset=False,
                                force_all_finite='allow-nan')

        n_samples, _ = X.shape
        
        Xt = np.hstack((X, np.ones((n_samples, 1))))
        Xc = np.hstack((X, np.zeros((n_samples, 1))))

        if is_classifier(self):
            return (self.estimator.predict_proba(Xt, **kwargs)[:, 1]  
                    - self.estimator.predict_proba(Xc, **kwargs))[:, 1] 
        return (self.estimator.predict(Xt, **kwargs) 
                - self.estimator.predict(Xc, **kwargs))


class _TLearner(BaseEstimator, MetaEstimatorMixin, UpliftMixin, metaclass=ABCMeta):
    def __init__(self, 
                 *,
                 estimator=None,
                 estimator_t=None,
                 estimator_c=None):
        self.estimator = estimator
        self.estimator_t = estimator_t
        self.estimator_c = estimator_c

    def fit(self, X, y, w, **fit_params):
        X, y = self._validate_data(X, y, force_all_finite='allow-nan')
        _, w = check_X_y(X, w, force_all_finite='allow-nan')

        if self.estimator is None:
            estimator_t = clone(self.estimator_t)
            estimator_c = clone(self.estimator_c)
        else:
            estimator_t = clone(self.estimator)
            estimator_c = clone(self.estimator)

        if estimator_t is None or estimator_c is None:
            raise ValueError('Invalid parameters estimators')
        if (is_classifier(self) != is_classifier(estimator_t)
            or is_classifier(self) != is_classifier(estimator_c)):
            raise ValueError('Estimators must be same type')

        estimator_t.fit(X[w == 1], y[w == 1],
                        **fit_params.get('estimator_t', {}))
        estimator_c.fit(X[w == 0], y[w == 0],
                        **fit_params.get('estimator_c', {}))

        self.estimator = None
        self.estimator_t = estimator_t
        self.estimator_c = estimator_c

        return self        

    def predict(self, X, **kwargs):
        check_is_fitted(self)
        X = self._validate_data(X, reset=False,
                                force_all_finite='allow-nan')
        
        if is_classifier(self):
            pred_t = (self.estimator_t
                      .predict_proba(X, **kwargs.get('estimator_t', {}))[:, 1])
            pred_c = (self.estimator_c
                      .predict_proba(X, **kwargs.get('estimator_c', {}))[:, 1])
        else:
            pred_t = (self.estimator_t
                      .predict(X, **kwargs.get('estimator_t', {})))
            pred_c = (self.estimator_c
                      .predict(X, **kwargs.get('estimator_c', {})))

        return pred_t - pred_c


class _XLearner(BaseEstimator, MetaEstimatorMixin, UpliftMixin, metaclass=ABCMeta):
    def __init__(self,
                 *,
                 estimator=None,
                 samples_beta: float = 0.5,
                 estimator_alpha=None,
                 estimator_alpha_t=None,
                 estimator_alpha_c=None,
                 estimator_beta=None,
                 estimator_beta_t=None,
                 estimator_beta_c=None,
                 propencity: float = 0.5,
                 samples_p: float = None,
                 estimator_p=None,
                 random_state: int = None):
        self.estimator = estimator
        self.samples_beta = samples_beta
        self.estimator_alpha = estimator_alpha
        self.estimator_alpha_t = estimator_alpha_t
        self.estimator_alpha_c = estimator_alpha_c
        self.estimator_beta = estimator_beta
        self.estimator_beta_t = estimator_beta_t
        self.estimator_beta_c = estimator_beta_c
        self.propencity = propencity
        self.samples_p = samples_p
        self.estimator_p = estimator_p
        self.random_state = random_state

    def fit(self, X, y, w, **fit_params):
        X, y = self._validate_data(X, y, force_all_finite='allow-nan')
        _, w = check_X_y(X, w, force_all_finite='allow-nan')

        if self.estimator is None:
            if self.estimator_alpha is None:
                estimator_alpha_t = clone(self.estimator_alpha_t)
                estimator_alpha_c = clone(self.estimator_alpha_c)
            else:
                estimator_alpha_t = clone(self.estimator_alpha)
                estimator_alpha_c = clone(self.estimator_alpha)
            
            if self.estimator_beta is None:
                estimator_beta_t = clone(self.estimator_beta_t)
                estimator_beta_c = clone(self.estimator_beta_c)
            else:
                estimator_beta_t = clone(self.estimator_beta)
                estimator_beta_c = clone(self.estimator_beta)
        else:
            estimator_alpha_t = clone(self.estimator) 
            estimator_alpha_c = clone(self.estimator) 
            estimator_beta_t = clone(self.estimator) 
            estimator_beta_c = clone(self.estimator) 

        if (estimator_alpha_t is None or estimator_alpha_c is None
            or estimator_beta_t is None or estimator_beta_c is None):
            raise ValueError('Invalid parameters estimators')
        if (is_classifier(self) != is_classifier(estimator_alpha_t)
            or is_classifier(self) != is_classifier(estimator_alpha_c)
            or is_classifier(self) != is_classifier(estimator_beta_t)
            or is_classifier(self) != is_classifier(estimator_beta_c)):
            raise ValueError('Estimators must be same type')

        random_state = check_random_state(self.random_state)

        if self.propencity is None:
            propencity = None
            if self.estimator is None:
                estimator_p = clone(self.estimator_p)
            else:
                estimator_p = clone(self.estimator)
            if estimator_p is None:
                raise ValueError('Invalid parameters estimators')
            if not is_classifier(estimator_p):
                raise ValueError('Estimator for propencity must be classifier')

            samples_p = check_scalar(self.samples_p,
                                     'samples_p',
                                     float,
                                     min_val=0, max_val=1,
                                     include_boundaries='neither')
            
            (X_alpha_beta, X_p,
             y_alpha_beta, _,
             w_alpha_beta, w_p,) = train_test_split(X, y, w,
                                                    test_size=samples_p,
                                                    stratify=w,
                                                    random_state=random_state)

            estimator_p.fit(X_p, w_p)
        else:
            propencity = check_scalar(self.propencity,
                                     'propencity',
                                      (int, float),
                                      min_val=0, max_val=1,)
            estimator_p = None
            samples_p = None

            X_alpha_beta, y_alpha_beta, w_alpha_beta = X, y, w

        samples_beta = check_scalar(self.samples_beta,
                                    'samples_beta',
                                    float,
                                    min_val=0, max_val=1,
                                    include_boundaries='neither')

        (X_alpha, X_beta,
         y_alpha, y_beta,
         w_alpha, w_beta,) = train_test_split(X_alpha_beta,
                                              y_alpha_beta,
                                              w_alpha_beta,
                                              test_size=samples_beta,
                                              stratify=w_alpha_beta,
                                              random_state=random_state)

        estimator_alpha_t.fit(X_alpha[w_alpha == 1], y_alpha[w_alpha == 1],
                              **fit_params.get('estimator_alpha_t', {}))
        estimator_alpha_c.fit(X_alpha[w_alpha == 0], y_alpha[w_alpha == 0],
                              **fit_params.get('estimator_alpha_c', {}))

        if is_classifier(self):
            dt_beta = (y_beta[w_beta == 1] 
                       - estimator_alpha_c.predict_proba(X_beta[w_beta == 1])[:, 1])
            dc_beta = (estimator_alpha_t.predict_proba(X_beta[w_beta == 0])[:, 1] 
                       - y_beta[w_beta == 0])
        else:
            dt_beta = (y_beta[w_beta == 1] 
                       - estimator_alpha_c.predict(X_beta[w_beta == 1]))
            dc_beta = (estimator_alpha_t.predict(X_beta[w_beta == 0]) 
                       - y_beta[w_beta == 0])

        estimator_beta_t.fit(X_beta[w_beta == 1], dt_beta,
                             **fit_params.get('estimator_beta_t', {}))
        estimator_beta_c.fit(X_beta[w_beta == 0], dc_beta,
                             **fit_params.get('estimator_beta_c', {}))
        
        self.estimator = None
        self.estimator_alpha = None
        self.estimator_alpha_t = estimator_alpha_t
        self.estimator_alpha_c = estimator_alpha_c
        self.estimator_beta = None
        self.estimator_beta_t = estimator_beta_t
        self.estimator_beta_c = estimator_beta_c
        self.propencity = propencity
        self.samples_p = samples_p
        self.estimator_p = estimator_p

        return self

    def predict(self, X, **kwargs):
        check_is_fitted(self)
        X = self._validate_data(X, reset=False,
                                force_all_finite='allow-nan')
        
        if self.propencity is None:
            propencity = (self.estimator_p
                          .predict_proba(X, **kwargs.get('estimator_p', {}))[:, 1])
        else:
            propencity = np.full(X.shape[0], self.propencity)

        if is_classifier(self):
            pred_t = (self.estimator_beta_t
                      .predict_proba(X, **kwargs.get('estimator_t', {}))[:, 1])
            pred_c = (self.estimator_beta_c
                      .predict_proba(X, **kwargs.get('estimator_c', {}))[:, 1])
        else:
            pred_t = (self.estimator_beta_t
                      .predict(X, **kwargs.get('estimator_t', {})))
            pred_c = (self.estimator_beta_c
                      .predict(X, **kwargs.get('estimator_c', {})))

        return propencity * pred_c + (1 - propencity) * pred_t


class _RLearner(BaseEstimator, MetaEstimatorMixin, UpliftMixin, metaclass=ABCMeta):
    pass


class SRegressor(_SLearner, RegressorMixin):
    pass


class SClassifier(_SLearner, ClassifierMixin):
    pass


class TRegressor(_TLearner, RegressorMixin):
    pass


class TClassifier(_TLearner, ClassifierMixin):
    pass


class XRegressor(_XLearner, RegressorMixin):
    pass


class XClassifier(_XLearner, ClassifierMixin):
    pass


class RRegressor(_RLearner, RegressorMixin):
    pass


class RClassifier(_RLearner, ClassifierMixin):
    pass
