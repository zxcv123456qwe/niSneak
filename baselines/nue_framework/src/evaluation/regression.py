from evaluation import MetricScorer
from .formulas import mar, sa, sd, sdar, effect_size, mmre, mdmre, pred25, pred40
from baseline import MARP0

class MAR(MetricScorer):
    
    def setConstants(self):
        self.name = "mar"
        self.problem = "regression"
        self.greater_is_better = False
        self.lo = 0
        self.hi = 20000 # Not really, but upped bound is infinity
        self.baseline = MARP0
        self.unifeature = False
        self.composite = None
    
    def _score_func(self, y_true, y_pred, X=None, estimator=None):
        return mar(self, y_true, y_pred)

class SDAR(MetricScorer):
    
    def setConstants(self):
        self.name = "sdar"
        self.problem = "regression"
        self.greater_is_better = False
        self.lo = 0
        self.hi = 200000 # Not really, but upped bound is infinity
        self.baseline = MARP0
        self.unifeature = False
        self.composite = None
    
    def _score_func(self, y_true, y_pred, X=None, estimator=None):
        return sdar(self, y_true, y_pred)

class SA(MetricScorer):
    
    def setConstants(self):
        self.name = "sa"
        self.problem = "regression"
        self.greater_is_better = True
        self.lo = 0
        self.hi = 1 # Not really, but upped bound is infinity
        self.baseline = MARP0
        self.unifeature = False
        self.composite = None
    
    def _score_func(self, y_true, y_pred, X=None, estimator=None):
        return sa(self, y_true, y_pred)

class SD(MetricScorer):
    
    def setConstants(self):
        self.name = "sd"
        self.problem = "regression"
        self.greater_is_better = True
        self.lo = 0
        self.hi = 1 # Not really, but upped bound is infinity
        self.baseline = MARP0
        self.unifeature = False
        self.composite = None
    
    def _score_func(self, y_true, y_pred, X=None, estimator=None):
        return sd(self, y_true, y_pred)

class EFFECTSIZE(MetricScorer):
    def setConstants(self):
        self.name = "effect size"
        self.problem = "regression"
        self.greater_is_better = True
        self.lo = 0
        self.hi = 1 # Not really, but upped bound is infinity
        self.baseline = MARP0
        self.unifeature = False
        self.composite = None
    
    def _score_func(self, y_true, y_pred, X=None, estimator=None):
        return effect_size(self, y_true, y_pred)

class MMRE(MetricScorer):
    def setConstants(self):
        self.name = "mmre"
        self.problem = "regression"
        self.greater_is_better = False
        self.lo = 0
        self.hi = 20000 # Not really, but upped bound is infinity
        self.baseline = MARP0
        self.unifeature = False
        self.composite = None
    
    def _score_func(self, y_true, y_pred, X=None, estimator=None):
        return mmre(self, y_true, y_pred)
    
class MdMRE(MetricScorer):
    def setConstants(self):
        self.name = "mdmre"
        self.problem = "regression"
        self.greater_is_better = False
        self.lo = 0
        self.hi = 20000 # Not really, but upped bound is infinity
        self.baseline = MARP0
        self.unifeature = False
        self.composite = None
    
    def _score_func(self, y_true, y_pred, X=None, estimator=None):
        return mdmre(self, y_true, y_pred)

class PRED25(MetricScorer):
    def setConstants(self):
        self.name = "pred25"
        self.problem = "regression"
        self.greater_is_better = True
        self.lo = 0
        self.hi = 1 # Not really, but upped bound is infinity
        self.baseline = MARP0
        self.unifeature = False
        self.composite = None
    
    def _score_func(self, y_true, y_pred, X=None, estimator=None):
        return pred25(self, y_true, y_pred)

class PRED40(MetricScorer):
    def setConstants(self):
        self.name = "pred40"
        self.problem = "regression"
        self.greater_is_better = True
        self.lo = 0
        self.hi = 1 # Not really, but upped bound is infinity
        self.baseline = MARP0
        self.unifeature = False
        self.composite = None
    
    def _score_func(self, y_true, y_pred, X=None, estimator=None):
        return pred40(self, y_true, y_pred)