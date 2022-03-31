from abc import ABCMeta, abstractmethod

import numpy as np


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

    def initialize(self, X, y, w):
        self.X = X
        self.y = y
        self.w = w

        self.n_features = self.X.shape[1]
        self.constant_features = list()
        self.n_constant_features = 0
        for i in range(self.n_features):
            if len(np.unique(self.X[:, i])) == 1:
                self.constant_features.append(i)
                self.n_constant_features += 1

    def node_value(self, idx):
        return self.criterion.node_value(self.y[idx], self.w[idx])

    def thresholds(self, Xi):
        thresholds = np.unique(Xi)
        if np.isnan(thresholds).any():
            thresholds = thresholds[~np.isnan(thresholds)]
        return thresholds

    @abstractmethod
    def split(self, idx, impurity):
        best_split = tuple([None, None, None, None])
        best_improvement = -np.inf

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
                left_idx = idx & (Xi <= threshold)
                right_idx = idx & (Xi > threshold)

                nt_left = (self.w[left_idx] == 1).sum()
                nc_left = (self.w[left_idx] == 0).sum()

                nt_right = (self.w[right_idx] == 1).sum()
                nc_right = (self.w[right_idx] == 0).sum()

                if ((nt_left + nc_left) < self.min_samples_leaf
                    or nt_left < self.min_samples_leaf_treated
                    or nc_left < self.min_samples_leaf_control):
                    continue
                if ((nt_right + nc_right) < self.min_samples_leaf
                    or nt_right < self.min_samples_leaf_treated
                    or nc_right < self.min_samples_leaf_control):
                    continue

                left = self.node_value(left_idx)
                right = self.node_value(right_idx)

                improvement = self.criterion.improvement(impurity,
                                                         nt_left + nc_left,
                                                         left[0],
                                                         nt_right + nc_right,
                                                         right[0],)

                if improvement > best_improvement:
                    best_improvement = improvement
                    best_split = (feature,
                                  threshold,
                                  tuple([left_idx]) + left,
                                  tuple([right_idx]) + right, )

            n_visited_features += 1
        
        return best_split + tuple([best_improvement])


class BestSplitter(Splitter):
    def split(self, idx, impurity):
        return super().split(idx, impurity)


class FastSplitter(Splitter):
    def thresholds(self, Xi):
        thresholds = super().thresholds(Xi)
        if len(thresholds) > 10:
            percentiles = np.percentile(thresholds,
                                        [3, 5, 10, 20, 30, 50, 70, 80, 90, 95, 97])
        else:
            percentiles = np.percentile(thresholds, [10, 50, 90])
        return np.unique(percentiles)

    def split(self, idx, impurity):
        return super().split(idx, impurity)
