from abc import ABCMeta, abstractmethod

import numpy as np


_epsilon = np.finfo('double').eps


class Criterion(metaclass=ABCMeta):
    def __init__(self, groups) -> None:
        self.groups = groups

    @abstractmethod
    def children(self, y, w):
        pass

    @abstractmethod
    def summary(self, 
                value_left, value_right,
                size_left, size_right):
        pass


class DeltaDeltaP(Criterion):
    def children(self, y, w):
        sizes = [(w == g).sum() for g in self.groups]

        value = 0
        for group in self.groups:
            yg = y[(w == group) | (w == 0)]
            wg = w[(w == group) | (w == 0)]
            wg[wg == group] = 1

            value += sizes[group - 1] * np.abs(yg[wg == 1].mean() - yg[wg == 0].mean())
        value /= sum(sizes)

        return value

    def summary(self,
                value_left, value_right,
                size_left, size_right):
        delta = np.abs(value_left - value_right)            
        return delta.sum()
