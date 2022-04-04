from abc import ABCMeta, abstractmethod

import numpy as np


_epsilon = np.finfo('double').eps


class Criterion(metaclass=ABCMeta):
    def __init__(self, groups) -> None:
        self.groups = groups

    @abstractmethod
    def value(self, y, w):
        pass

    @abstractmethod
    def gain(self,
             value_parent,
             value_left, value_right,
             size_left, size_right):
        pass


class DeltaDeltaP(Criterion):
    def value(self, y, w):
        sizes = [(w == g).sum() for g in self.groups]

        value = 0
        for group in self.groups:
            yg = y[(w == group) | (w == 0)]
            wg = w[(w == group) | (w == 0)]
            wg[wg == group] = 1

            value += sizes[group - 1] * np.abs(yg[wg == 1].mean() - yg[wg == 0].mean())
        value /= sum(sizes)

        return value

    def gain(self,
             value_parent,
             value_left, value_right,
             size_left, size_right):
        delta = np.abs(value_left - value_right)            
        return delta.sum()


class _DivergenceCriterion(Criterion):
    @abstractmethod
    def formula(self, pt, pc):
        pass

    def value(self, y, w):
        sizes = [(w == g).sum() for g in self.groups]
        pt = np.zeros(2)
        pc = np.zeros(2)

        pc[0] = (y[w == 0] == 0).sum() / (w == 0).sum()
        pc[1] = (y[w == 0] == 1).sum() / (w == 0).sum()

        divergence = 0
        for group in self.groups:
            pt[0] = (y[w == group] == 0).sum() / (w == 0).sum()
            pt[1] = (y[w == group] == 1).sum() / (w == 0).sum()

            divergence += sizes[group - 1] * self.formula(pt, pc)
        divergence /= sum(sizes)

        return divergence

    def gain(self,
             value_parent,
             value_left,
             value_right,
             size_left,
             size_right,):
        value = size_left * value_left + size_right * value_right
        value /= size_left + size_right

        return value - value_parent


class KLDivergence(_DivergenceCriterion):
    def formula(self, pt, pc):
        pc += _epsilon
        pt += _epsilon
        return np.sum(pt * np.log(pt / pc))


class EuclideanDivergence(_DivergenceCriterion):
    def formula(self, pt, pc):
        return np.sum((pt - pc)**2)


class Chi2Divergence(_DivergenceCriterion):
    def formula(self, pt, pc):
        pc += _epsilon
        return np.sum((pt - pc)**2 / pc)
