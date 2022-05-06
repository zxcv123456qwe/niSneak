from sklearn.base import BaseEstimator
from evaluation import GenericMetric


class BaseSPLModel( BaseEstimator ):

    def score(self):
        pass
    
    def n_parameters(self):
        return len(self._get_param_names())
    
    def n_objectives(self):
        return len(self.obj_weights)
    
    def get_scorings(self):
        m = []
        for n, s, hi, lo in zip(self.obj_names, self.obj_weights, self.obj_hi, self.obj_lo):
            m += [ (n, GenericMetric(n, s, lo, hi)) ]
        return dict(m)
    
    