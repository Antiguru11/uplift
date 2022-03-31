from abc import ABCMeta, abstractmethod

import numpy as np


_epsilon = np.finfo('double').eps


class Tree():
    def __init__(self):
        self.reset()

    def reset(self):
        self.nodes = list()
        self.leaf_ids = list()

    def add_node(self,
                 parent,
                 is_left, 
                 is_leaf,
                 feature,
                 threshold,
                 impurity,
                 n_treatment_samples,
                 n_control_samples,
                 uplift, ) -> int:
        node_id = len(self.nodes)
        self.nodes.append([node_id,
                           parent, None, None,
                           feature, threshold, impurity,
                           n_treatment_samples, n_control_samples, uplift])

        if parent is not None:
            if is_left:
                self.nodes[parent][2] = node_id
            else:
                self.nodes[parent][3] = node_id

        if is_leaf:
            self.leaf_ids.append(node_id)

        return node_id

    def apply(self, X) -> np.ndarray:
        uplift = np.full(X.shape[0], np.nan)

        for leaf_id in self.leaf_ids:
            mask = np.full(X.shape[0], True, dtype=bool)

            parent_id = self.nodes[leaf_id][1]
            child_id = leaf_id
            while parent_id is not None:
                mask &= self._apply_node(X, parent_id, child_id)
                parent_id, child_id = self.nodes[parent_id][1], parent_id

            uplift[mask] = self.nodes[leaf_id][-1]
        
        return uplift

    def _apply_node(self, X, parent_id, child_id):
        is_left = self.nodes[parent_id][2] == child_id
        Xi = X[:, self.nodes[parent_id][4]]
        threshold = self.nodes[parent_id][5]

        mask = Xi <= threshold if is_left else Xi > threshold

        if np.isnan(Xi).any():
            other_child_id = self.nodes[parent_id][2 if is_left else 3]

            a_impurity = self.nodes[child_id][6]
            b_impurity = self.nodes[other_child_id][6]

            if (a_impurity < b_impurity
                or (np.abs(a_impurity - _epsilon) <= b_impurity and is_left)):
                mask |= np.isnan(Xi)

        return mask


class TreeBuilder(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self,
                 splitter,
                 max_depth: int,
                 min_samples_split: int,
                 min_samples_leaf: int,
                 min_samples_leaf_treated: int,
                 min_samples_leaf_control: int,
                 max_leaf_nodes: int, ):
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_leaf_treated = min_samples_leaf_treated
        self.min_samples_leaf_control = min_samples_leaf_control
        self.max_leaf_nodes = max_leaf_nodes

    @abstractmethod
    def build(self, tree, X, y, w):
        pass


class DepthFirstTreeBuilder(TreeBuilder):
    def __init__(self,
                 splitter,
                 max_depth,
                 min_samples_split,
                 min_samples_leaf,
                 min_samples_leaf_treated,
                 min_samples_leaf_control):
        super().__init__(splitter,
                         max_depth,
                         min_samples_split,
                         min_samples_leaf,
                         min_samples_leaf_treated,
                         min_samples_leaf_control,
                         None,)

    def build(self, tree, X, y, w):
        tree.reset()
        self.splitter.initialize(X, y, w)

        stack = list()
        stack.append((np.full_like(y, True, dtype=bool), 
                      0,
                      None,
                      True,
                      np.inf,
                      None, 
                      None,
                      None, ))
        is_first = True
        while len(stack) != 0:
            item = stack.pop()

            idx = item[0]
            depth = item[1]
            parent = item[2]
            is_left = item[3]
            impurity = item[4]
            n_treatment_samples = item[5]
            n_control_samples = item[6]
            uplift = item[7]

            if is_first:
                is_first == False

                node_value = self.splitter.node_value(idx)

                impurity = node_value[0]
                n_treatment_samples = node_value[1]
                n_control_samples = node_value[2]
                uplift = node_value[3]

            is_leaf = (depth >= self.max_depth 
                       or (n_treatment_samples + n_control_samples) < self.min_samples_split
                       or (n_treatment_samples + n_control_samples) < self.min_samples_leaf
                       or n_treatment_samples < self.min_samples_leaf_treated
                       or n_control_samples < self.min_samples_leaf_control
                       or np.abs(impurity) <= _epsilon)

            feature = None
            threshold = None
            if not is_leaf:
                (feature,
                 threshold,
                 left,
                 right,
                 improvement) = self.splitter.split(idx, impurity)
                
                is_leaf = is_leaf or improvement <= _epsilon
            
            node_id = tree.add_node(parent, is_left, is_leaf,
                                    feature, threshold, impurity,
                                    n_treatment_samples, n_control_samples, uplift)

            if is_leaf:
                continue
            
            stack.append((left[0],
                          depth + 1,
                          node_id,
                          True, 
                          left[1],
                          left[2],
                          left[3],
                          left[4],))
            stack.append((right[0],
                          depth + 1,
                          node_id,
                          False, 
                          right[1],
                          right[2],
                          right[3],
                          right[4],))


class BestFirstTreeBuilder(TreeBuilder):
    def __init__(self,
                 splitter,
                 max_depth,
                 min_samples_split,
                 min_samples_leaf,
                 min_samples_leaf_treated,
                 min_samples_leaf_control,
                 max_leaf_nodes, ):
        super().__init__(splitter,
                         max_depth,
                         min_samples_split,
                         min_samples_leaf,
                         min_samples_leaf_treated,
                         min_samples_leaf_control,
                         max_leaf_nodes,)

    def build(self, tree, X, y, w):
        tree.reset()
        self.splitter.initialize(X, y, w)

        stack = list()
        stack.append((np.full_like(y, True, dtype=bool), 
                      0,
                      None,
                      True,
                      np.inf,
                      None, 
                      None,
                      None, ))
        is_first = True
        max_split_nodes = self.max_leaf_nodes - 1
        while len(stack) != 0 and max_split_nodes >= 0:
            next_id = np.array([si[4] for si in stack]).argmin()
            item = stack.pop(next_id)

            idx = item[0]
            depth = item[1]
            parent = item[2]
            is_left = item[3]
            impurity = item[4]
            n_treatment_samples = item[5]
            n_control_samples = item[6]
            uplift = item[7]

            if is_first:
                is_first == False

                node_value = self.splitter.node_value(idx)

                impurity = node_value[0]
                n_treatment_samples = node_value[1]
                n_control_samples = node_value[2]
                uplift = node_value[3]

            is_leaf = (depth >= self.max_depth
                       or max_split_nodes <= 0
                       or (n_treatment_samples + n_control_samples) < self.min_samples_split
                       or (n_treatment_samples + n_control_samples) < self.min_samples_leaf
                       or n_treatment_samples < self.min_samples_leaf_treated
                       or n_control_samples < self.min_samples_leaf_control
                       or np.abs(impurity) <= _epsilon)

            if not is_leaf:
                (feature,
                 threshold,
                 left,
                 right,
                 improvement) = self.splitter.split(idx, impurity)
                
                is_leaf = is_leaf or improvement <= _epsilon
            
            node_id = tree.add_node(parent, is_left, is_leaf,
                                    None if is_leaf else feature,
                                    None if is_leaf else threshold,
                                    impurity,
                                    n_treatment_samples, n_control_samples, uplift)

            if is_leaf:
                continue
            
            stack.append((left[0],
                          depth + 1,
                          node_id,
                          True, 
                          left[1],
                          left[2],
                          left[3],
                          left[4],))
            stack.append((right[0],
                          depth + 1,
                          node_id,
                          False, 
                          right[1],
                          right[2],
                          right[3],
                          right[4],))
            
            max_split_nodes -= 1
