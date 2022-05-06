from validation import Cross_Validation
from sklearn.model_selection import train_test_split

# TODO Change KFold to
# Use sklearn kfold
# Store indexes instead of tables
class KFold(Cross_Validation):
    
    def __init__(self, predict, train_size = 0.9, n_iter = 20, baseline = None):
        self.predict = predict
        self.train_size = train_size
        self.test_size = 1 - train_size
        self.n_iter = n_iter
        self.X_train = []
        self.Y_train = []
        self.X_test = []
        self.Y_test = []
        self.baseline = baseline
        self.base_mar = []
        self.base_std = []
    
    
    def split(self, df):
        self.X_train = []
        self.Y_train = []
        self.X_test = []
        self.Y_test = []
        self.base_mar = []
        self.base_std = []
        
        X = df[ df.columns.difference([self.predict]) ]
        Y = df[self.predict]
        
        for i in range(0, self.n_iter):
            X_tr, X_te, Y_tr, Y_te = train_test_split(X, Y, train_size = self.train_size, test_size = self.test_size)
            self.X_train.append(X_tr)
            self.Y_train.append(Y_tr)
            self.X_test.append(X_te)
            self.Y_test.append(Y_te)
            mar, std = self.baseline.fit(Y_te) if self.baseline is not None else (None, None)
            self.base_mar.append( mar )
            self.base_std.append( std )
        
    def __iter__(self):
        self.i = 0
        return self
    
    def __next__(self):
        if self.i >= self.n_iter:
            raise StopIteration
        X_tr = self.X_train[self.i]
        Y_tr = self.Y_train[self.i]
        X_te = self.X_test[self.i]
        Y_te = self.Y_test[self.i]
        b_mar = self.base_mar[self.i]
        b_std = self.base_std[self.i]
        self.i += 1
        return X_tr, X_te, Y_tr, Y_te, b_mar, b_std