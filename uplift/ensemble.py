from abc import ABCMeta, abstractmethod

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import is_classifier
from sklearn.utils import check_X_y, check_scalar, check_random_state
from sklearn.ensemble import BaseEnsemble
from sklearn.utils.validation import check_is_fitted

from .base import UpliftMixin, RegressorMixin, ClassifierMixin
from .tree import DecisionTreeRegressor, DecisionTreeClassifier


def _generate_indexes(source_size, sample_size):
    return np.random.choice(np.arange(source_size),
                            sample_size,
                            replace=True)


def _parallel_fit(tree, X, y, w, max_samples, max_features):
    n_samples, n_features = X.shape

    samples_idx = _generate_indexes(n_samples, max_samples)
    if max_features < n_features: 
        features_idx = _generate_indexes(n_features, max_features)
    else:
        features_idx = np.arange(n_features)

    tree.fit(X[samples_idx, :][:, features_idx], y[samples_idx], w[samples_idx])

    return tree


def _parallel_predict(tree, X):
    return tree.predict(X)


class BaseForest(BaseEnsemble, UpliftMixin, metaclass=ABCMeta):
    def __init__(self,
                 base_estimator,
                 *,
                 n_estimators: int,
                 estimator_params: tuple,
                 samples_by_estimator: int,
                 features_by_estimator: int,
                 n_jobs: int,
                 verbose: int,
                 random_state: int):
        super().__init__(base_estimator=base_estimator,
                         n_estimators=n_estimators,
                         estimator_params=estimator_params)
        self.samples_by_estimator = samples_by_estimator
        self.features_by_estimator = features_by_estimator
        self.n_jobs = n_jobs 
        self.verbose = verbose
        self.random_state = random_state 

    def fit(self, X, y, w):
        X, y = self._validate_data(X, y, force_all_finite='allow-nan')
        _, w = check_X_y(X, w, force_all_finite='allow-nan')

        n_samples, self.n_features_in_ = X.shape
        
        if isinstance(self.samples_by_estimator, int):
            n_samples_by_estimator = check_scalar(self.samples_by_estimator,
                                                   'features_by_estimator',
                                                    int,
                                                    min_val=1,
                                                    max_val=n_samples)
        elif isinstance(self.samples_by_estimator, float):
            n_samples_by_estimator = check_scalar(self.samples_by_estimator,
                                                   'features_by_estimator',
                                                    float,
                                                    min_val=0,
                                                    max_val=1,
                                                    include_boundaries='right')
            n_samples_by_estimator = int(n_samples_by_estimator * n_samples)
        elif self.samples_by_estimator is None:
            n_samples_by_estimator = n_samples
        else:
            raise ValueError('Invalid value for samples_by_estimator')
        
        if isinstance(self.features_by_estimator, int):
            n_features_by_estimator = check_scalar(self.features_by_estimator,
                                                   'features_by_estimator',
                                                    int,
                                                    min_val=1,
                                                    max_val=self.n_features_in_)
        elif isinstance(self.features_by_estimator, float):
            n_features_by_estimator = check_scalar(self.features_by_estimator,
                                                   'features_by_estimator',
                                                    float,
                                                    min_val=0,
                                                    max_val=1,
                                                    include_boundaries='right')
            n_features_by_estimator = int(n_features_by_estimator * self.n_features_in_)
        elif self.features_by_estimator is None:
            n_features_by_estimator = self.n_features_in_
        else:
            raise ValueError('Invalid value for features_by_estimator')
        
        random_state = check_random_state(self.random_state)

        self._validate_estimator()
        self.estimators_ = list()

        trees = [self._make_estimator(append=False, random_state=random_state) 
                 for _ in range(self.n_estimators)]

        trees = Parallel(n_jobs=self.n_jobs,
                         verbose=self.verbose,
                         prefer='threads',)(delayed(_parallel_fit)(tree,
                                                                   X, y, w,
                                                                   n_samples_by_estimator,
                                                                   n_features_by_estimator,)
                                            for tree in trees)

        self.estimators_.extend(trees)

        return self

    def predict(self, X):
        check_is_fitted(self)

        X = self._validate_data(X, reset=False,
                                force_all_finite='allow-nan')

        preds = Parallel(n_jobs=self.n_jobs,
                         verbose=self.verbose,
                         prefer='threads',)(delayed(_parallel_predict)(tree, X)
                                            for tree in self.estimators_)

        return np.array(preds).mean(axis=0)


class RandomForestRegressor(BaseForest, RegressorMixin):
    def __init__(self,
                 *,
                 n_estimators: int = 100,
                 criterion: str = 'delta',
                 splitter: str = 'fast',
                 max_depth: int = None,
                 min_samples_split: int = 40,
                 min_samples_leaf: int = 20,
                 min_samples_leaf_treated: int = 10,
                 min_samples_leaf_control: int = 10,
                 max_features: int = None,
                 max_leaf_nodes: int = None,
                 samples_by_estimator: int = None,
                 features_by_estimator: int = None,
                 n_jobs: int = None,
                 verbose: int = None,
                 random_state: int = None):
        super().__init__(base_estimator=DecisionTreeRegressor(),
                         n_estimators=n_estimators,
                         estimator_params=('criterion',
                                           'splitter',
                                           'max_depth',
                                           'min_samples_split',
                                           'min_samples_leaf',
                                           'min_samples_leaf_treated',
                                           'min_samples_leaf_control',
                                           'max_features',
                                           'max_leaf_nodes',
                                           'random_state',),
                         samples_by_estimator=samples_by_estimator,
                         features_by_estimator=features_by_estimator,
                         n_jobs=n_jobs,
                         verbose=verbose,
                         random_state=random_state)
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_leaf_treated = min_samples_leaf_treated
        self.min_samples_leaf_control = min_samples_leaf_control
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes


class RandomForestClassifier(BaseForest, ClassifierMixin):
    pass
