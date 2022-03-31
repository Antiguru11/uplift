class UpliftMixin:
    _estimator_type = "uplift"

    def score(self, X, y, w):
        from .metrics import uplift_at_k

        return uplift_at_k(self.predict(X), y, w)

    def _more_tags(self):
        return {"requires_y": True}


class ClassifierMixin:
    _estimator_type = 'classifier'


class RegressorMixin:
    _estimator_type = 'regressor'
