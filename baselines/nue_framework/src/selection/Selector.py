from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class ColumnSelector(BaseEstimator, TransformerMixin):
    
    def __init__(self, columns):
        self.columns = columns
        
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.columns]
    

class NumericalSelector(BaseEstimator, TransformerMixin):
    
    def __init__(self, nums = True):
        self.nums = nums
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        include = np.logical_xor([x.kind in 'buifc' for x in X.dtypes], not(self.nums))
        newX = X.iloc[:,include].copy()
        return newX
