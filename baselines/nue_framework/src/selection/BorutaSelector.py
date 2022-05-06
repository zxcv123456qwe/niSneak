from boruta import BorutaPy

class BorutaSelector(BorutaPy):
    def fit(self, X, y):
        super()._fit( X.values, y.values )
    
    def transform(self, X, weak=False):
        # sanity check
        try:
            self.ranking_
        except AttributeError:
            raise ValueError('You need to call the fit(X, y) method first.')

        if weak:
            indices = self.support_ + self.support_weak_
        else:
            indices = self.support_

        X = X.iloc[:, indices]
        
        return X
    
    def fit_transform(self, X, y, weak=False):
        self.fit(X, y)
        return self.transform(X, weak)