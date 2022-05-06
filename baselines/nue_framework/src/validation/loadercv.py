from sklearn.model_selection import BaseCrossValidator
import pandas as pd
import numpy as np

class LoaderCV(BaseCrossValidator):
    idx_dir = "idx/"
    
    def __init__(self, group = 0, offset = 0):
        self.group = group
        self.offset = offset
    
    def set_data(self, dataset_id):
        self.dataset_id = dataset_id
    
    def set_config(self, config_dir):
        self.config_dir = config_dir
    
    def get_n_splits(self, X=None, y=None, groups=None):
        return self.dataframe_test.shape[0]
    
    def split(self, X=None, y=None, groups=None):
        if type(self.group) != list:
            self.group = [self.group]
        for g in self.group:
            file_name = self.config_dir + self.idx_dir + self.dataset_id + "_test" + ( "_%d" % (g-1) if g != 0 else "" ) + ".csv"
            self.dataframe_test = pd.read_csv( file_name, header = None, index_col = False )
            file_name = self.config_dir + self.idx_dir + self.dataset_id + "_train" + ( "_%d" % (g-1) if g != 0 else "" ) + ".csv"
            self.dataframe_train = pd.read_csv( file_name, header = None, index_col = False )
            for (i, row_train), (_, row_test) in zip(self.dataframe_train.iterrows(), self.dataframe_test.iterrows()):
                train_idx = np.copy(row_train)
                train_idx = train_idx[~np.isnan(train_idx)]
                test_idx = np.copy(row_test)
                test_idx = test_idx[~np.isnan(test_idx)]
                if self.offset > 0:
                    self.offset -= 1
                else:
                    yield train_idx.astype("int64"), test_idx.astype("int64")
        
