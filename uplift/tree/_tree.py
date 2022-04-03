from abc import ABCMeta, abstractmethod
from turtle import right

import numpy as np


_epsilon = np.finfo('double').eps


class Tree():
    def __init__(self, n_groups):
        self.n_groups = n_groups
        self.reset()

    def reset(self):
        self.nodes = list()
        self.leaf_ids = list()

    def add_node(self,
                 parent,
                 is_left, 
                 is_leaf,
                 value,
                 best_value,
                 split,
                 stats) -> int:
        node_id = len(self.nodes)
        self.nodes.append([node_id, parent, None, None,
                           value, best_value, 
                           split[0], split[1],
                           stats[0], stats[1], stats[2]])

        if parent is not None:
            if is_left:
                self.nodes[parent][2] = node_id
            else:
                self.nodes[parent][3] = node_id

        if is_leaf:
            self.leaf_ids.append(node_id)

        return node_id

    def apply(self, X) -> np.ndarray:
        n_samples, _ = X.shape

        uplift = np.full((n_samples, self.n_groups), np.nan)

        for leaf_id in self.leaf_ids:
            mask = np.full(X.shape[0], True, dtype=bool)

            parent_id = self.nodes[leaf_id][1]
            child_id = leaf_id
            while parent_id is not None:
                mask &= self._apply_node(X, parent_id, child_id)
                parent_id, child_id = self.nodes[parent_id][1], parent_id

            uplift[mask, :] = np.array(self.nodes[leaf_id][-1])
        
        return uplift

    def _apply_node(self, X, parent_id, child_id):
        is_left = self.nodes[parent_id][2] == child_id
        feature, threshold = self.nodes[parent_id][6:8]

        Xi = X[:, feature]

        if np.isnan(threshold):
            mask = np.isnan(Xi) if is_left else ~np.isnan(Xi)
        else:
            mask = Xi <= threshold if is_left else Xi > threshold

            if np.isnan(Xi).any():
                other_child_id = self.nodes[parent_id][2 if is_left else 3]

                a_value = self.nodes[child_id][4]
                b_value = self.nodes[other_child_id][4]

                if (a_value > b_value
                    or (np.abs(a_value - b_value) <= _epsilon and is_left)):
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
    def build(self, tree, X, y, w, groups):
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

    def build(self, tree, X, y, w, groups):
        tree.reset()
        self.splitter.initialize(X, y, w, groups)

        stack = list()
        stack.append((np.full_like(y, True, dtype=bool), 
                      0,
                      None,
                      True,
                      None,
                      (None, None, None),))
        is_first = True
        while len(stack) != 0:
            item = stack.pop()
            best_value = None
            split = (None, None)

            idx, depth, parent, is_left, value, stats = item
            (n_treatments,
             n_control, 
             uplift, ) = stats

            if is_first:
                is_first == False

                (value,
                 n_treatments,
                 n_control,
                 uplift, ) = self.splitter.node_value(idx)

            is_leaf = (depth >= self.max_depth 
                       or (sum(n_treatments) + n_control) < self.min_samples_split
                       or (sum(n_treatments) + n_control) < self.min_samples_leaf
                       or min(n_treatments) < self.min_samples_leaf_treated
                       or n_control < self.min_samples_leaf_control)

            if not is_leaf:
                best_value, best_split = self.splitter.split(idx)

                (split,
                 (idx_left,
                  value_left,
                  stats_left,),
                 (idx_right,
                  value_right,
                  stats_right,)) = best_split
                
                is_leaf = is_leaf or best_value <= _epsilon
            
            node_id = tree.add_node(parent, is_left, is_leaf,
                                    value, best_value, split,
                                    (n_treatments, n_control, uplift))

            if is_leaf:
                continue
            
            stack.append((idx_left,
                          depth + 1,
                          node_id,
                          True, 
                          value_left,
                          stats_left,))
            stack.append((idx_right,
                          depth + 1,
                          node_id,
                          False, 
                          value_right,
                          stats_right,))


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

    def build(self, tree, X, y, w, groups):
        tree.reset()
        self.splitter.initialize(X, y, w, groups)

        stack = list()
        stack.append((np.full_like(y, True, dtype=bool), 
                      0,
                      None,
                      True,
                      None,
                      (None, None, None),))
        is_first = True
        max_split_nodes = self.max_leaf_nodes - 1
        while len(stack) != 0 and max_split_nodes >= 0:
            next_id = np.array([si[4] for si in stack]).argmax()
            item = stack.pop(next_id)

            best_value = None
            split = (None, None)

            idx, depth, parent, is_left, value, stats = item
            (n_treatments,
             n_control, 
             uplift, ) = stats

            if is_first:
                is_first == False

                (value,
                 n_treatments,
                 n_control,
                 uplift, ) = self.splitter.node_value(idx)

            is_leaf = (depth >= self.max_depth
                       or max_split_nodes <= 0
                       or (sum(n_treatments) + n_control) < self.min_samples_split
                       or (sum(n_treatments) + n_control) < self.min_samples_leaf
                       or min(n_treatments) < self.min_samples_leaf_treated
                       or n_control < self.min_samples_leaf_control)

            if not is_leaf:
                best_value, best_split = self.splitter.split(idx)

                (split,
                 (idx_left,
                  value_left,
                  stats_left,),
                 (idx_right,
                  value_right,
                  stats_right,)) = best_split
                
                is_leaf = is_leaf or best_value <= _epsilon
            
            node_id = tree.add_node(parent, is_left, is_leaf,
                                    value, best_value, split,
                                    (n_treatments, n_control, uplift))

            if is_leaf:
                continue
            
            stack.append((idx_left,
                          depth + 1,
                          node_id,
                          True, 
                          value_left,
                          stats_left,))
            stack.append((idx_right,
                          depth + 1,
                          node_id,
                          False, 
                          value_right,
                          stats_right,))
            
            max_split_nodes -= 1
