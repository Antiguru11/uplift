from abc import ABCMeta, abstractmethod

import numpy as np

from ._utils import group_stats


class Splitter(metaclass=ABCMeta):
    def __init__(self,
                 criterion,
                 min_samples_leaf,
                 min_samples_leaf_treated,
                 min_samples_leaf_control,
                 max_features,
                 random_state,):
        self.criterion = criterion
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_leaf_treated = min_samples_leaf_treated
        self.min_samples_leaf_control = min_samples_leaf_control
        self.max_features = max_features
        self.random_state = random_state

    def initialize(self, X, y, w, groups):
        self.X = X
        self.y = y
        self.w = w
        self.groups = groups

        self.n_features = self.X.shape[1]
        self.constant_features = list()
        self.n_constant_features = 0
        for i in range(self.n_features):
            if len(np.unique(self.X[:, i])) == 1:
                self.constant_features.append(i)
                self.n_constant_features += 1

    def node_value(self, idx):
        value = self.criterion.children(self.y[idx], self.w[idx])
        return tuple([value]) + group_stats(self.y[idx],
                                            self.w[idx],
                                            self.groups)

    def thresholds(self, Xi):
        return np.unique(Xi).tolist()

    @abstractmethod
    def split(self, idx):
        best_split = tuple([(None, None),
                            (None, None, (None, None, None)),
                            (None, None, (None, None, None)),])
        best_value = -np.inf

        features = [i 
                    for i in range(self.n_features) 
                    if i not in self.constant_features]
        np.random.shuffle(features)

        n_visited_features = 0
        while (n_visited_features < self.max_features
               and len(features) != 0):
            feature = features.pop()
            
            Xi = self.X[:, feature]

            thresholds = self.thresholds(Xi[idx])

            if len(thresholds) == 1:
                n_visited_features += 1
                continue

            for threshold in thresholds:
                if np.isnan(threshold):
                    idx_left = idx & np.isnan(Xi)
                    idx_right = idx & ~np.isnan(Xi)
                else:
                    idx_left = idx & (Xi <= threshold)
                    idx_right = idx & (Xi > threshold)

                (nts_left,
                 nc_left, 
                 uplift_left) = group_stats(self.y[idx_left],
                                            self.w[idx_left],
                                            self.groups)
                
                (nts_right,
                 nc_right, 
                 uplift_right) = group_stats(self.y[idx_right],
                                             self.w[idx_right],
                                             self.groups)

                if ((sum(nts_left) + nc_left) < self.min_samples_leaf
                    or (sum(nts_right) + nc_right) < self.min_samples_leaf
                    or min(min(nts_left), min(nts_right)) < self.min_samples_leaf_treated
                    or min(nc_left, nc_right) < self.min_samples_leaf_control):
                    continue

                value_left = self.criterion.children(self.y[idx_left],
                                                     self.w[idx_left])
                value_right = self.criterion.children(self.y[idx_right],
                                                      self.w[idx_right])

                value = self.criterion.summary(value_left, value_right,
                                               idx_left.sum(), idx_right.sum())

                if value > best_value:
                    best_value = value
                    best_split = ((feature, threshold),
                                  tuple([idx_left,
                                         value_left,
                                         (nts_left,
                                          nc_left,
                                          uplift_left)]),
                                  tuple([idx_right,
                                         value_right,
                                         (nts_right,
                                          nc_right,
                                          uplift_right)]))

            n_visited_features += 1
        
        return best_value, best_split


class BestSplitter(Splitter):
    def split(self, idx):
        return super().split(idx)


class FastSplitter(Splitter):
    def thresholds(self, Xi):
        have_nan = np.isnan(Xi).any()

        values = Xi[~np.isnan(Xi)]
        if len(values) > 10:
            thresholds = np.percentile(values,
                                       [3, 5, 10, 20, 30, 50, 70, 80, 90, 95, 97])
        else:
            thresholds = np.percentile(values, [10, 50, 90])
        
        thresholds = np.unique(thresholds).tolist()
        if have_nan:
            thresholds = [np.nan] + thresholds

        return thresholds

    def split(self, idx):
        return super().split(idx)
