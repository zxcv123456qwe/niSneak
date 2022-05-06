from sklearn.base import BaseEstimator
import numpy as np
from utils import ps
from sklearn.neighbors import NearestNeighbors
from pandas.api.types import is_numeric_dtype
import pandas as pd

def sample_neighbors(df, knn, n_samples = 2):
    """
    Returns a random point and some of its neighbors
    Input:
        - df,Dataframe: Data
        - knn,object: NearestNeighbor object from sklearn fitted on df
        - n_samples: amount of neighbors to return
    Output:
        List with the sampled datapoint first and its neghbors following it.
    """
    sample = np.random.randint( 0, df.shape[0] - 1 )
    sample = df.iloc[sample:sample+1, :]
    neigh = knn.kneighbors(sample, n_samples, return_distance=False)[0]
    neigh = df.iloc[neigh,:]
    return neigh

class WFO(BaseEstimator, ps):
    """
    Class:
        FairSmoteSelector
    Description:
        Minority class resampling
        Based on https://github.com/yrahul3910/raise/blob/master/raise_utils/transforms/wfo.py
    Attributes:
        features,list: Unused
    """
    
    def __init__(self, features, *, r0 = 0.0, r1 = 0.3, ri = 0.03):
        self.features = features
        self.r0 = r0
        self.r1 = r1
        self.ri = ri
    
    def fit(self, X, y, **fit_params):
        pass
    
    def transmute(self, X, y):
        self._unique = y.unique()
        self._counts = [ np.sum( y == i ) for i in self._unique ]
        self._majority = self._unique[ np.argmax( self._counts ) ]
        # print(self._unique, self._counts, self._majority)
        
        df = X.copy()
        df["target"] = y
        new_df = df.copy()
        
        for val in self._unique:
            if val == self._majority:
                continue
            idx = np.where( df["target"] == val )[0]
            frac = 1.0 * len(idx) / df.shape[0]
            
            for row in X.iloc[idx,:].itertuples(index=False):
                for i, r in enumerate( np.arange(self.r0, self.r1, self.ri) ):
                    for _ in range( int((1 / frac) / pow(2.0, i)) ):
                        row1, row2 = [], []
                        for x in row:
                            if type(x) in [str, np.character, pd.Series]:
                                row1.append( x )
                                row2.append( x )
                            else:
                                row1.append( x - r )
                                row2.append( x + r )
                        row1.append( val )
                        row2.append( val )
                        new_df = new_df.append( pd.DataFrame([row1], columns = new_df.columns) )
                        new_df = new_df.append( pd.DataFrame([row2], columns = new_df.columns) )
        
        X, y = new_df[new_df.columns.difference(["target"])], new_df["target"]
        return X, y
        
    def fit_transmute(self, X, y = None, **fit_args):
        self.fit(X, y, **fit_args)
        return self.transmute(X, y)

if __name__ == "__main__":
    import pandas as pd
    d = {
        "feature1" : [2.5, 2.7, 2.2, 2.6, 2.0, 2.1, 2.3, 5.4, 5.2, 5.1, 4.2, 4.3, 5.1, 4.9],
        "feature2" : [0,0,0,0,1,1,1,0,0,0,1,1,1,1],
        "feature3" : ["a", "b", "a", "b", "a", "b", "a", "b","a", "b","a", "b","a", "b"],
        "target" : [0,0,0,0,0,0,0,0,0,0,1,1,1,1],
    }
    
    df = pd.DataFrame.from_dict( d )
    X, y = df[df.columns.difference(["target"])], df["target"]
    print(X)
    print(y)
    
    protected_features = ["protected1"]
    
    fs = WFO(None, r0 = 0.0, r1 = 0.3, ri = 0.03)
    fs.fit(X, y)
    print(fs)
    X_new, y_new = fs.transmute(X, y)
    
    df = X_new.copy()
    df["target"] = y_new
    print(df)
    print(df.shape)
        