from sklearn.base import (BaseEstimator as _BaseEstimator,
                          MultiOutputMixin,)
from sklearn.utils import check_consistent_length

from .utils import check_w


class BaseEstimator(_BaseEstimator, MultiOutputMixin):
    def _validate_data(self,
                       X,
                       y = "no_validation",
                       w = "no_validation",
                       reset=True,
                       **check_params):
        out = super()._validate_data(X, y,
                                     reset=reset,
                                     **check_params)
        if not (isinstance(w, str) and w == 'no_validation'):
            w, groups = check_w(w)
            check_consistent_length(X, y, w)

            if reset:
                self.groups = groups
                self.n_groups = len(self.groups)

            out = tuple(list(out) + [w])
        
        return out


class UpliftMixin:
    _estimator_type = "uplift"


class ClassifierMixin:
    _estimator_type = 'classifier'

    def score(self, X, y, w):
        if self.n_groups < 2:
            from .metrics import uplift_at_k

            return uplift_at_k(self.predict(X), y, w)
        else:
            from .metrics import uplift_at_k_macro

            return uplift_at_k_macro(self.predict(X), y, w)


    def _more_tags(self):
        return {"requires_y": True}


class RegressorMixin:
    _estimator_type = 'regressor'

    def score(self, X, y, w):
        if self.n_groups < 2:
            from .metrics import uplift_at_k

            return uplift_at_k(self.predict(X), y, w)
        else:
            from .metrics import uplift_at_k_avg

            return uplift_at_k_avg(self.predict(X), y, w)

    def _more_tags(self):
        return {"requires_y": True}
