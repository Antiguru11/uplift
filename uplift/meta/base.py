from abc import ABCMeta, abstractmethod

import numpy as np
from sklearn.base import (MetaEstimatorMixin,
                          is_classifier,
                          clone,)
from sklearn.utils import check_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted

from ..base import BaseEstimator


class BaseLearner(BaseEstimator,
                  MetaEstimatorMixin,
                  metaclass=ABCMeta):
    def __init__(self,
                 estimator,
                 *,
                 estimators_params: list,
                 propencity: bool = False,
                 propencity_score=None,
                 propencity_estimator=None,
                 random_state: int = None):
        self.estimator = estimator
        self.estimators_params = estimators_params
        self.propencity = propencity
        self.propencity_score = propencity_score
        self.propencity_estimator = propencity_estimator
        self.random_state = random_state

    def _make_estimators(self):
        if self.estimator is not None:
            for e in self.estimators_params:
                setattr(self, e, clone(self.estimator))

        if len(self.estimators_params) != 0:
            self.estimator = None
        else:
            self.estimators_params = ('estimator',)
            
    @abstractmethod
    def _check_params(self):
        params = dict()

        for e in self.estimators_params:
            if getattr(self, e) is None:
                raise ValueError(f'Estimator {e} is None')
            if is_classifier(self) != is_classifier(getattr(self, e)):
                raise ValueError(f'Estimator {e} must be '
                                 + ('classifier' if is_classifier(self) else 'regressor'))

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

    def fit(self, X, y, w, **fit_params):
        X, y, w = self._validate_data(X, y, w, 
                                      reset=True,
                                      force_all_finite=self._get_tags()['allow_nan'])

        self._make_estimators()

        params = self._check_params()
        params['fit_params'] = fit_params
        
        self.estimators = [tuple() for i in range(self.n_groups)]
        if self.propencity_estimator is not None:
            self.p_estimators = [tuple() for i in range(self.n_groups)]
        for group in self.groups:
            Xg = X[(w == group) | (w == 0)]
            yg = y[(w == group) | (w == 0)]
            wg = w[(w == group) | (w == 0)]
            wg[wg == group] = 1

            if self.propencity_estimator is not None:
                self._fit_propencity(group, Xg, wg, 
                                     **fit_params.get('propencity_estimator', {}))
            
            self._fit_group(group,
                            Xg, yg, wg,
                            **params)

        return self
    
    @abstractmethod
    def _fit_group(self, group, X, y, w, **kwargs):
        pass

    def _fit_propencity(self, group, X, w, **fit_params):
        (X, X_calib,
         w, w_calib,) = train_test_split(X, w,
                                         test_size=0.5,
                                         stratify=w,
                                         random_state=self.random_state)

        estimator = clone(self.propencity_estimator)
        estimator.fit(X, w, **fit_params)

        estimator_calib = LogisticRegression(random_state=self.random_state)
        estimator_calib.fit(estimator.predict_proba(X_calib)[:, 1].reshape(-1, 1),
                            w_calib)

        self.p_estimators[group - 1] = (estimator, estimator_calib)

    def predict(self, X, **kwargs):
        check_is_fitted(self)
        self._validate_data(X, reset=False, 
                            force_all_finite=self._get_tags()['allow_nan'])

        n_samples, _ = X.shape

        preds = np.full((n_samples, self.n_groups), np.nan)
        for group in self.groups:
            preds[:, group - 1] = self._predict_group(group, X, **kwargs)
        
        if self.n_groups == 1:
            return preds.reshape(-1)
        return preds
    
    @abstractmethod
    def _predict_group(self, group, X, **kwargs):
        pass

    def _predict_propencity(self, group, X, **kwargs):
        if self.propencity:
            if self.propencity_score is None:
                estimator, estimator_calib = self.p_estimators[group - 1]

                pred = estimator.predict_proba(X)[:, 1].reshape(-1, 1)
                return estimator_calib.predict_proba(pred)[:, 1]
            else:
                return np.full(X.shape[0], self.propencity_score[group - 1])
        else:
            raise ValueError('Propencity is not supported')
        