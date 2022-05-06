from sklearn.pipeline import FeatureUnion, _transform_one, _fit_transform_one
from joblib import Parallel, delayed
import numpy as np
import pandas as pd

class FeatureJoin(FeatureUnion):
    def transform(self, X):
        Xs = Parallel(n_jobs=self.n_jobs)(
            delayed(_transform_one)(trans, X, None, weight)
            for name, trans, weight in self._iter())
        if not Xs:
            # All transformers are None
            return np.zeros((X.shape[0], 0))
        return pd.concat(Xs, axis=1)
    
    def fit_transform(self, X,  y=None, **fit_params):
        results = self._parallel_func(X, y, fit_params, _fit_transform_one)
        if not results:
            # All transformers are None
            return np.zeros((X.shape[0], 0))

        Xs, transformers = zip(*results)
        self._update_transformer_list(transformers)
        return pd.concat(Xs, axis=1)
    