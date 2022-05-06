from validation import Cross_Validation
from sklearn.model_selection import RepeatedKFold

class NxM(Cross_Validation):
    
    # n_splits = N
    # n_repeats = M
    def __init__(self, predict, n_splits = 10 , n_repeats = 10):
        self.predict = predict
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.train_index = []
        self.test_index = []
        self.X = None
        self.Y = None
    
    def split(self, df):
        self.train_index = []
        self.test_index = []
        
        self.X = df[ df.columns.difference([self.predict]) ]
        self.Y = df[self.predict]
        
        rskf = RepeatedKFold(n_splits=self.n_splits, n_repeats=self.n_repeats)
        for train_index, test_index in rskf.split(self.X, self.Y):
            self.train_index.append(train_index)
            self.test_index.append(test_index)  
            Y_te = self.Y[test_index]
        
    def __iter__(self):
        self.i = 0
        return self
    
    def __next__(self):
        if self.i >= self.n_splits * self.n_repeats:
            raise StopIteration
        tr_i = self.train_index[ self.i ]
        te_i = self.test_index[ self.i ]
        X_tr = self.X.iloc[tr_i,:]
        Y_tr = self.Y.iloc[tr_i]
        X_te = self.X.iloc[te_i,:]
        Y_te = self.Y.iloc[te_i]
        self.i += 1
        return X_tr, X_te, Y_tr, Y_te
    
    def set_dataframe(self, df):
        self.X = df[ df.columns.difference([self.predict]) ]
        self.Y = df[self.predict]