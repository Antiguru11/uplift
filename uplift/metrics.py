from sklearn.utils import check_array, check_consistent_length, check_scalar


def uplift_at_k(uplift, y, w,
                *,
                k: float = 0.3,
                strategy: str = 'overall') -> float:
    uplift = check_array(uplift, ensure_2d=False)
    y = check_array(y, ensure_2d=False)
    w = check_array(w, ensure_2d=False)

    check_consistent_length(uplift, y, w)
    k = check_scalar(k, 'k', float, min_val=1e-5, max_val=1.0)

    order = uplift.argsort()[::-1]
    if strategy == 'overall':
        n = int(len(uplift) * k)

        yt = y[order][:n][w[order][:n] == 1]
        yc = y[order][:n][w[order][:n] == 0]

        return yt.mean() - yc.mean()
    else:
        raise NotImplementedError
