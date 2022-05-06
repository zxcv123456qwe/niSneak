from sklearn.base import BaseEstimator, TransformerMixin

class DummySelector(BaseEstimator, TransformerMixin):
    def fit(self, X, Y, **fit_params):
        return self
    def transform(self, X):
        return X
    def get_params(self, **params):
        return {}