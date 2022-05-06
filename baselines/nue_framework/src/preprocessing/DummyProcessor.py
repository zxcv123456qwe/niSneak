class DummyProcessor:
    def __init__(self, **kwargs):
        pass
    def fit(self, X, y = None, **fit_args):
        pass
    def transmute(self, X, y = None):
        return X, y
    def fit_transmute(self, X, y = None, **fit_args):
        self.fit(X, y, **fit_args)
        return self.transmute(X, y)