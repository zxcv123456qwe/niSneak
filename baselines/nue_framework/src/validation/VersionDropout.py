from sklearn.model_selection import BaseCrossValidator
import numpy as np
import pandas as pd

class VersionDropout(BaseCrossValidator):
    
    def __init__(self, n_repeats = 10, dropout = 0.3):
        self.n_repeats = n_repeats
        self.dropout = dropout
    
    # We dont implement this one
    def _iter_test_indices(self, X=None, y=None, groups=None):
        pass
    
    def split(self, X, y=None, groups=None):
        rows = X.shape[0]
        latest = sorted(np.unique(X["version"]))[-1]
        train_idx_base = []
        for i, j in zip(range(rows), X["version"] != latest):
            if j: train_idx_base += [i]
        test_idx_base = []
        for i, j in zip(range(rows), X["version"] == latest):
            if j: test_idx_base += [i]
        
        for i in range(self.n_repeats):
            n = len(train_idx_base)
            n_samples = int(np.ceil( n * (1-self.dropout) ))
            np.random.shuffle( train_idx_base )
        
            yield train_idx_base[0:n_samples], test_idx_base
    
    def get_n_splits(self, X, y=None, groups=None):
        return self.n_repeats


if __name__ == "__main__":
    versions = np.reshape(np.array( [1,1,1,2,2,2,2,3,3,3] ), (10,1))
    data = np.reshape(np.zeros(100), (10, 10))
    X = pd.DataFrame( np.append( versions, data, axis = 1 ), columns = ["version"] + [i for i in range(10)] )
    y =  pd.DataFrame(np.zeros(10))
    
    for i, (train_index, test_index) in enumerate(VersionDropout(10, 0.2).split(X, y)):
        print( "iter: %d\ntraining: %s\ntesting: %s\n" % (i, train_index, test_index) )