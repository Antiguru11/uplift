import numpy as np


def check_w(w):
    from sklearn.utils import check_array

    w = check_array(w, dtype=int, ensure_2d=False)

    groups = np.unique(w)
    if len(groups) < 2:
        raise ValueError('Treatment vector must have atleast two group')
    if 0 not in groups:
        raise ValueError('Treatment vector must have zero value for control group')
    if len(set(groups).difference(set(range(len(groups))))) != 0:
        raise ValueError('Groups in the treatment vector must be in sequential order')

    return w, sorted(groups[groups != 0].tolist())
