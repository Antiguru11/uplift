from functools import partial

import numpy as np
from sklearn.utils import check_array, check_consistent_length, check_scalar

from .utils import check_w


def uplift_at_k(uplift, y, w,
                *,
                k: float = 0.3,
                strategy: str = 'overall',) -> float:
    uplift = check_array(uplift, ensure_2d=False)
    y = check_array(y, ensure_2d=False)
    w, groups = check_w(w)

    if len(groups) > 1:
        raise ValueError('Not supported')

    check_consistent_length(uplift, y, w)

    k = check_scalar(k, 'k', float, min_val=1e-5, max_val=1.0)

    order = uplift.argsort()[::-1]
    if strategy == 'overall':
        n = int(len(uplift) * k)

        yt = y[order][:n][w[order][:n] != 0]
        yc = y[order][:n][w[order][:n] == 0]

        return yt.mean() - yc.mean()
    else:
        raise NotImplementedError


def _average(*, metric, method='avg'):
    def _metric(uplift, y, w, **kwargs):
        uplift = check_array(uplift)
        y = check_array(y, ensure_2d=False)
        w, groups = check_w(w)

        values = list()
        for group in groups:
            ug = uplift[(w == group) | (w == 0), group - 1]
            yg = y[(w == group) | (w == 0)]
            wg = w[(w == group) | (w == 0)]
            wg[wg == group] = 1
            ng = len(wg)
            values.append((metric(ug, yg, wg, **kwargs), ng))
        
        if method == 'avg':
            return sum([v for v,_ in values]) / len(groups)
        else:
            raise ValueError('Invalid parameter method')
    return _metric


uplift_at_k_avg = _average(metric=uplift_at_k, method='avg')
