import numbers
from abc import ABCMeta, abstractmethod

from sklearn.base import BaseEstimator, is_classifier
from sklearn.utils import check_X_y, check_random_state, check_scalar

from . import _criterion, _splitter
from ._tree import Tree, DepthFirstTreeBuilder, BestFirstTreeBuilder
from ..base import UpliftMixin, ClassifierMixin, RegressorMixin


_criteria_clf = {}
_criteria_reg = {'delta': _criterion.Delta,}

_splitters = {'best': _splitter.BestSplitter,
              'fast': _splitter.FastSplitter,}


class BaseDecisionTree(BaseEstimator, UpliftMixin, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self,
                 *,
                 criterion: str,
                 splitter: str,
                 max_depth: int,
                 min_samples_split: int,
                 min_samples_leaf: int,
                 min_samples_leaf_treated: int,
                 min_samples_leaf_control: int,
                 max_features: int,
                 max_leaf_nodes: int,
                 random_state: int, ):
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_leaf_treated = min_samples_leaf_treated
        self.min_samples_leaf_control = min_samples_leaf_control
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.random_state = random_state

    def fit(self, X, y, w):
        X, y = self._validate_data(X, y, force_all_finite='allow-nan')
        _, w = check_X_y(X, w, force_all_finite='allow-nan')

        _, self.n_features_in_ = X.shape
        is_classification = is_classifier(self)

        max_depth = np.iinfo(np.int32).max if self.max_depth is None else self.max_depth
        max_leaf_nodes = -1 if self.max_leaf_nodes is None else self.max_leaf_nodes

        min_samples_split = check_scalar(self.min_samples_split,
                                         'min_samples_split',
                                         int,
                                         min_val=1,)
        min_samples_leaf_treated = check_scalar(self.min_samples_leaf_treated,
                                         'min_samples_leaf',
                                         int,
                                         min_val=1,)
        min_samples_leaf_control = check_scalar(self.min_samples_leaf_control,
                                         'min_samples_leaf',
                                         int,
                                         min_val=1,)
        if self.min_samples_leaf is None:
            min_samples_leaf = (self.min_samples_leaf_treated 
                                + self.min_samples_leaf_control)
        else:
            min_samples_leaf = check_scalar(self.min_samples_leaf,
                                            'min_samples_leaf',
                                            int,
                                            min_val=1,)
            if (min_samples_leaf < min_samples_leaf_treated 
                or min_samples_leaf < min_samples_leaf_control):
                raise ValueError("Invalid value for min_samples_leaf")
        
        min_samples_split = max(min_samples_split, 2 * min_samples_leaf)

        if isinstance(self.max_features, str):
            if self.max_features == "auto":
                if is_classification:
                    max_features = max(1, int(np.sqrt(self.n_features_in_)))
                else:
                    max_features = self.n_features_in_
            elif self.max_features == "sqrt":
                max_features = max(1, int(np.sqrt(self.n_features_in_)))
            elif self.max_features == "log2":
                max_features = max(1, int(np.log2(self.n_features_in_)))
            else:
                raise ValueError("Invalid value for max_features")
        elif self.max_features is None:
            max_features = self.n_features_in_
        elif isinstance(self.max_features, numbers.Integral):
            max_features = self.max_features
        else:  # float
            if self.max_features > 0.0:
                max_features = max(1, int(self.max_features * self.n_features_in_))
            else:
                max_features = 0

        random_state = check_random_state(self.random_state)

        if is_classification:
            criterion = _criteria_clf[self.criterion]()
        else:
            criterion = _criteria_reg[self.criterion]()

        splitter = _splitters[self.splitter](criterion,
                                             min_samples_leaf,
                                             min_samples_leaf_treated,
                                             min_samples_leaf_control,
                                             max_features,
                                             random_state)

        self.tree_ = Tree()
        if max_leaf_nodes < 0:
            builder = DepthFirstTreeBuilder(splitter,
                                            max_depth,
                                            min_samples_split,
                                            min_samples_leaf,
                                            min_samples_leaf_treated,
                                            min_samples_leaf_control,)
        else:
            builder = BestFirstTreeBuilder(splitter,
                                           max_depth,
                                           min_samples_split,
                                           min_samples_leaf,
                                           min_samples_leaf_treated,
                                           min_samples_leaf_control,
                                           max_leaf_nodes,)

        builder.build(self.tree_, X, y, w)

        return self

    def predict(self, X):
        X = self._validate_data(X, reset=False,
                                force_all_finite='allow-nan')
        return self.tree_.apply(X)

    def _more_tags(self):
        return {'allow_nan': True}


class DecisionTreeRegressor(RegressorMixin, BaseDecisionTree):
    def __init__(self,
                 *,
                 criterion: str = 'delta',
                 splitter: str = 'best',
                 max_depth: int = None,
                 min_samples_split: int = 40,
                 min_samples_leaf: int = 20,
                 min_samples_leaf_treated: int = 10,
                 min_samples_leaf_control: int = 10,
                 max_features: int = None,
                 max_leaf_nodes: int = None,
                 random_state: int = None):
        super().__init__(criterion=criterion,
                         splitter=splitter,
                         max_depth=max_depth,
                         min_samples_split=min_samples_split,
                         min_samples_leaf=min_samples_leaf,
                         min_samples_leaf_treated=min_samples_leaf_treated,
                         min_samples_leaf_control=min_samples_leaf_control,
                         max_features=max_features,
                         max_leaf_nodes=max_leaf_nodes,
                         random_state=random_state)


class DecisionTreeClassifier(ClassifierMixin, BaseDecisionTree):
    pass
