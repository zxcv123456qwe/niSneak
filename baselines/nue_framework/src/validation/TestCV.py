from sklearn.model_selection import BaseCrossValidator
import pandas as pd
import numpy as np

class TestCV(BaseCrossValidator):
    idx_dir = "idx/"
    
    def __init__(self):
        pass
    
    def set_data(self, dataset_id):
        self.dataset_id = dataset_id
    
    def get_n_splits(self, X=None, y=None, groups=None):
        return 1
    
    def _iter_test_indices(self, X=None, y=None, groups=None):
        path = self.idx_dir + self.dataset_id + "_test.csv"
        idx_test = pd.read_csv( path, header = None, index_col = False )
        res = idx_test[~np.isnan(idx_test.iloc[0,:])]
        yield res.astype("int64")
        
