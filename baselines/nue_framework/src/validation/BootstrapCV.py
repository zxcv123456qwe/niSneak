from sklearn.model_selection import BaseCrossValidator
import numpy as np
import pandas as pd

class BootstrapCV(BaseCrossValidator):
    
    def __init__(self, n_bootstraps = 100):
        self.n_bootstraps = n_bootstraps
    
    # We dont implement this one
    def _iter_test_indices(self, X=None, y=None, groups=None):
        pass
    
    def split(self, X, y=None, groups=None):
        for i in range(self.n_bootstraps):
            n = X.shape[0]
            idx = [i for i in range(0, n)]
            train_index = np.random.choice(idx, n, True)
        
            test_index = [ i for i in idx if i not in train_index ]
        
            yield train_index, test_index
    
    def get_n_splits(self, X, y=None, groups=None):
        return self.n_bootstraps

if __name__ == "__main__":
    X = pd.DataFrame(np.reshape(np.zeros(100), (10, 10)))
    y =  pd.DataFrame(np.zeros(10))
    
    for i, (train_index, test_index) in enumerate(BootstrapCV(10).split(X, y)):
        print( "iter: %d\ntraining: %s\ntesting: %s\n" % (i, train_index, test_index) )