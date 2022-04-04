import numpy as np


def group_stats(y, w, groups):
    uts = list()
    nts = list()

    nc = (w == 0).sum()
    if nc == 0:
        yc = 0
    else:
        yc = y[w == 0].mean()

    for group in groups:
        ng = (w == group).sum()
        if ng == 0:
            uts.append(-yc)
        else:
            uts.append(y[w == group].mean() - yc)
        nts.append(ng)

    return tuple(nts), nc, tuple(uts)
