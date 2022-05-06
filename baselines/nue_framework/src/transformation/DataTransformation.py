from calendar import c
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, FunctionTransformer
from utils import ps
import sklearn
import numpy as np

class DataTransformation(ps):
    
    def __init__(self, name = "None", dt_class = None, parameters = {}):
        self.name = name
        self.dt_class = dt_class
        self.parameters = parameters

class OneHotEncoding(OneHotEncoder):
    def transform(self, X):
        res = super().transform(X)
        if sklearn.__version__[0] == "0":
            return pd.DataFrame(res, columns=self.get_feature_names(X.columns))
        else:
            return pd.DataFrame(res, columns=self.get_feature_names_out(X.columns))

class StandardScaling(StandardScaler):
    def transform(self, X, copy=None):
        return pd.DataFrame( super().transform(X), columns=X.columns )

class MinMaxScaling(MinMaxScaler):
    def transform(self, X, copy=None):
        return pd.DataFrame( super().transform(X), columns=X.columns )

class LogMinMaxScaling(MinMaxScaling):
    def transform(self, X, copy=None):
        return super().transform( FunctionTransformer(np.log1p).fit_transform(X) )


