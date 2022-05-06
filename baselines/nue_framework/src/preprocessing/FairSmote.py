from sklearn.base import BaseEstimator
import numpy as np
from utils import ps
from sklearn.neighbors import NearestNeighbors
from pandas.api.types import is_numeric_dtype
import pandas as pd

def data_slice(df, feature_vals):
    """
    Yields the sub-datasets of all possible permutations of feature values
    Input:
        - df,Dataframe: Data
        - feature_vals,dict: Dictionary of feature name and possible values
    Output:
        Yields each sub-dataset
    """
    if len(feature_vals) == 0:
        yield df
    else:
        feature_vals = feature_vals.copy()
        feat = list(feature_vals.keys())[0]
        values = feature_vals.pop( feat, None )
        if values is None:
            yield from data_slice( df, feature_vals )
        else:
            for v in values:
                new_df = df[df[feat] == v]
                yield from data_slice( new_df, feature_vals )

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

class FairSmoteSelector(BaseEstimator, ps):
    """
    Class:
        FairSmoteSelector
    Description:
        Proposed by Chakraborty et al. Bias in Machine Learning Software: Why? How? What to Do?
        Fairness method for rebalancing datapoints.
        For privileged/unprivileged groups and favorable/unfavorable outcomes.
        Generates artificial data points so these groups are balanced.
    Attributes:
        features,list: List of names of protected features. We assume 0 is unprivileged.
    """
    
    def __init__(self, features, *, mutation_amount = 0.8, crossover_frequency = 0.8):
        self.features = features
        self.mutation_amount = mutation_amount
        self.crossover_frequency = crossover_frequency
    
    def fit(self, X, y, **fit_params):
        self._n_features = len(self.features)
        self._predict_val = { "target" : y.unique() }
        self._feature_val = dict([ (f, X[ f ].unique() ) for f in self.features ])
        self._joint_val = {**self._predict_val, **self._feature_val}
    
    def transmute(self, X, y):
        df = X.copy()
        df["target"] = y
        new_df = df.copy()
        
        # Calculate size of largest group
        n = max( [ sub.shape[0] for sub in data_slice(df, self._joint_val) ] )
        
        # Repeat for each possible group
        for sub in data_slice(df, self._joint_val):
            # We cannot generate new population if we dont have enough samples
            if sub.shape[0] >= 3:
                to_gen = n - sub.shape[0] # Calculate elements to generate to reach max
                knn = NearestNeighbors(n_neighbors=3, algorithm='auto').fit(sub)
                
                for i in range(to_gen):
                    new_candidate = {}
                    neigh = sample_neighbors(sub, knn, 3)
                    parent = neigh.iloc[0]
                    for c in df.columns:
                        col = df[c]
                        new_val = parent[c]
                        options = [ n[c] for i, n in neigh.iterrows() ]
                        # With 1 - cr%, generate a new value
                        if np.random.random() > self.crossover_frequency:
                            if is_numeric_dtype( col ) and sorted(df[c].unique()) == [0, 1]: # Numeric, non-binary
                                new_val += self.mutation_amount * ( options[1] - options[2] )
                            else: # String and binary
                                new_val = np.random.choice( options )
                        new_candidate[c] = [new_val]
                    new_candidate = pd.DataFrame.from_dict(new_candidate)
                    new_df = pd.concat( [new_df, new_candidate] )
            else:
                raise Exception("Insufficient data in a group.")
        
        X, y = new_df[new_df.columns.difference(["target"])], new_df["target"]
        return X, y
        
    def fit_transmute(self, X, y = None, **fit_args):
        self.fit(X, y, **fit_args)
        return self.transmute(X, y)

if __name__ == "__main__":
    import pandas as pd
    d = {
        "feature1" : [2.5, 2.7, 2.2, 2.6, 2.0, 2.1, 2.3, 5.4, 5.2, 5.1, 4.2, 4.3, 5.1, 4.9],
        "protected1" : [0,0,0,0,1,1,1,0,0,0,1,1,1,1],
        "target" : [0,0,0,0,0,0,0,1,1,1,1,1,1,1],
    }
    
    df = pd.DataFrame.from_dict( d )
    X, y = df[df.columns.difference(["target"])], df["target"]
    print(X)
    print(y)
    
    protected_features = ["protected1"]
    
    fs = FairSmoteSelector(protected_features)
    fs.fit(X, y)
    print(fs)
    X_new, y_new = fs.transmute(X, y)
    
    df = X_new.copy()
    df["target"] = y_new
    print(df)
        