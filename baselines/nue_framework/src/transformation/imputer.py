import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.base import BaseEstimator, TransformerMixin

class FillImputer(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass
        
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if X.size > 0:
            new_X = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value="missing").fit_transform(X)
            return pd.DataFrame( new_X, columns = X.columns )
        else:
            return X

class SimplerImputer(SimpleImputer):
    def transform(self, X):
        return pd.DataFrame( super().transform(X), columns = X.columns )

class KNNImputerDF(KNNImputer):
    def transform(self, X):
        return pd.DataFrame( super().transform(X), columns = X.columns )
