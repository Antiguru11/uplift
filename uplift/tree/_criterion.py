from abc import ABCMeta, abstractmethod

import numpy as np


_epsilon = np.finfo('double').eps


class Criterion(metaclass=ABCMeta):
    @abstractmethod
    def node_value(self, y, w):
        pass

    @abstractmethod
    def improvement(self,
                    parent_impurity,
                    n_left_samples,
                    left_impurity,
                    n_right_samples,
                    right_impurity):
        pass


class Delta(Criterion):
    def node_value(self, y, w):
        nt = (w == 1).sum()
        nc = (w == 0).sum()

        yt = y[w == 1]
        yc = y[w == 0]

        uplift = yt.mean() - yc.mean() + _epsilon

        return ((yt.std() + yc.std()) / np.abs(uplift), nt, nc, uplift)

    def improvement(self,
                    parent_impurity,
                    n_left_samples,
                    left_impurity,
                    n_right_samples,
                    right_impurity):
        return (parent_impurity
                - (n_left_samples * left_impurity + n_right_samples * right_impurity) 
                   / (n_left_samples + n_right_samples))
