import numpy as np
from scipy.stats.mstats import gmean
from scipy.stats import spearmanr
from baseline import MARP0
from baseline import MDARP0
from baseline import MARP0LOO

class Evaluation:
    
    def __init__(self, metrics = [], baseline = None, multiobj = [], reference = []):
        self.metrics = metrics
        self.baseline = baseline
        self.multiobj = multiobj
        self.reference = reference
    
    def get_function(self, name):
        r = None
        if name.lower() == "mmre":
            r = self.mmre
        elif name.lower() == "mdmre":
            r = self.mdmre
        elif name.lower() == "pred25":
            r = lambda y_true, y_pred : self.pred(25, y_true, y_pred)
        if name.lower() == "mmre100":
            r = self.mmre100
        elif name.lower() == "mdmre100":
            r = self.mdmre100
        elif name.lower() == "pred25100":
            r = lambda y_true, y_pred : self.pred100(25, y_true, y_pred)
        elif name.lower() == "mae" or name.lower() == "mar":
            r = self.mar
        elif name.lower() == "mdae" or name.lower() == "mdar":
            r = self.mdar
        elif name.lower() == "sdae" or name.lower() == "sdar":
            r = self.sdar
        elif name.lower() == "sa":
            r = self.sa
        elif name.lower() == "delta" or name.lower() == "effect size":
            r = self.effect_size
        elif name.lower() == "sd":
            r = self.sd
        elif name.lower() == "mbre":
            r = self.mbre
        elif name.lower() == "mibre":
            r = self.mibre
        elif name.lower() == "spearmancc":
            r = self.spearmancc
        elif name.lower() == "hv" or name.lower() == "hypervolume":
            r = self.hyper_volume
        return r
    
    def get_greater_is_better(self, name):
        r = None
        if name.lower() == "mmre" or name.lower() == "mmre100":
            r = False
        elif name.lower() == "mdmre" or name.lower() == "mdmre100":
            r = False
        elif name.lower() == "pred25" or name.lower() == "pred25100":
            r = True
        elif name.lower() == "mae" or name.lower() == "mar":
            r = False
        elif name.lower() == "mdae" or name.lower() == "mdar":
            r = False
        elif name.lower() == "sdae" or name.lower() == "sdar":
            r = True
        elif name.lower() == "sa":
            r = True
        elif name.lower() == "delta" or name.lower() == "effect size":
            r = True
        elif name.lower() == "sd":
            r = True
        elif name.lower() == "mbre":
            r = False
        elif name.lower() == "mibre":
            r = False
        elif name.lower() == "spearmancc":
            r = True
        elif name.lower() == "hv" or name.lower() == "hypervolume":
            r = True
        return r
    
    def evaluate(self, y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        res = {}
        
        for metric in self.metrics:
            r = None
            try:
                # Regular metrics
                r = self.get_function(metric.lower())(y_true, y_pred)
            except:
                # Baselines
                if metric.lower() == "marp0":
                    r = self.baseline.fit( y_true )[0]
                elif metric.lower() == "sp0":
                    r = self.baseline.fit( y_true )[1]
            res[metric] = r
        return res
    
    # Multi objective metric related functions
    def reference_point(self):
        # If we get a list the same size of all metrics, return that
        # If not, return -1 for all values
        # Useful for standardized techniques; i.e. sa and sd
        
        if len(self.reference) == len( self.multiobj ):
            reference = self.reference
        else:
            reference = [-1 for i in range(len(self.multiobj))]
        
        return dict( zip(self.multiobj, reference) )
    
    # Similar to evaluate, but based on multiobj list
    def evaluate_multiobj(self, y_true, y_pred):
        res = {}
        for metric in self.multiobj:
            r = self.get_function(metric)(y_true, y_pred)
            res[metric] = r
        return res
    
    ## Multi objective metrics
    
    # Single-point hyper_volume
    # Calculated as the product of all metrics
    def hyper_volume(self, y_true, y_pred):
        scores = list(self.evaluate_multiobj(y_true, y_pred).values())
        reference = list(self.reference_point().values())
        diff = np.subtract(scores, reference)
        
        # Dont accept negative values
        # If any are negative, they are converted to 0
        diff = [ max(0, v) for v in diff ]
        
        return np.product( diff )
    
    
    ## Single objective metrics
    
    # Effect size, also known as Glass's delta
    def effect_size(self, y_true, y_pred):
        if self.baseline is not None:
            base_c = self.baseline.fit( y_true )[0]
            base_s = self.baseline.fit( y_true )[1]
            if type(self.baseline) in [MARP0, MARP0LOO]:
                val = self.mar(y_true, y_pred)
            elif type(self.baseline) == MDARP0:
                val = self.mdar(y_true, y_pred)
            else: return None
            
            if base_s == 0:
                return np.abs( ( val - base_c + 1 ) / (base_s + 1) )
            else:
                return np.abs( ( val - base_c ) / base_s )
        return None
    
    # Standarized accuracy
    def sa(self, y_true, y_pred):
        if self.baseline is not None:
            base = self.baseline.fit( y_true )[0]
            if type(self.baseline) in [MARP0, MARP0LOO]:
                val = self.mar(y_true, y_pred)
            elif type(self.baseline) == MDARP0:
                val = self.mdar(y_true, y_pred)
            else: return None
            
            if base == 0:
                return 1 - ( (val + 1) / (base+1) )
            else:
                return 1 - ( val / base )
        return None
    
    # Absolute error, aka absolute resuidual or AR
    def ae_i(self, y_true, y_pred):
        return np.abs(y_true - y_pred)
    
    # Magnitude of relative error
    def mre(self, y_true, y_pred):
        return self.ae_i(y_true, y_pred) / y_true
    
    # Mean magnitude of relative error
    def mmre(self, y_true, y_pred):
        return np.mean(self.mre(y_true, y_pred))
    
    # Mean magnitude of relative error, as percentage
    def mmre100(self, y_true, y_pred):
        return self.mmre(y_true, y_pred) * 100
    
    # Median magnitude of relative error
    def mdmre(self, y_true, y_pred):
        return np.median(self.mre(y_true, y_pred))
    
    # Median magnitude of relative error, as percentage
    def mdmre100(self, y_true, y_pred):
        return self.mdmre(y_true, y_pred) * 100
    
    # PRED(X), usually PRED(25). % of predictions above 25% of the MRE
    def pred(self, n, y_true, y_pred):
        return np.sum( self.mre(y_true, y_pred) <= (n/100) ) / y_true.size
    
    # PRED(X), usually PRED(25). % of predictions above 25% of the MRE, as percentage
    def pred100(self, n, y_true, y_pred):
        return self.pred(n, y_true, y_pred) * 100
    
    # Mean absolute residual, aka mean absolute error or mae
    def mar(self, y_true, y_pred):
        return np.average(self.ae_i(y_true, y_pred))
    
    # Median absolute residual, aka median absolute error or mdae or mdae
    def mdar(self, y_true, y_pred):
        return np.median(self.ae_i(y_true, y_pred))
    
    # Standard deviation of absolute residual
    def sdar(self, y_true, y_pred):
        return np.std(self.ae_i(y_true, y_pred))
    
    # Standarized deviation
    # Based off the stability ratio
    # With respect to baseline
    def sd(self, y_true, y_pred):
        if self.baseline is not None:
            base = self.baseline.fit( y_true )[1]
            if type(self.baseline) in [MARP0, MDARP0, MARP0LOO]:
                val = self.sdar(y_true, y_pred)
            else: return None
            
            if base == 0:
                return 1 - ((val + 1) / (base + 1))
            else:
                return 1 - (val / base)
        return None
    
    # geometric mean of the absolute residual
    def gmar(self, y_true, y_pred):
        return gmean(self.ae_i(y_true, y_pred))
    
    # Balanced relative error
    def bre(self, y_true, y_pred):
        return np.divide(np.abs( y_true - y_pred ), np.minimum(y_true, y_pred) )
    
    # Mean balanced relative error
    def mbre(self, y_true, y_pred):
        return np.mean(self.bre(y_true, y_pred))
    
    # Inverse balanced relative error
    def ibre(self, y_true, y_pred):
        return np.divide(np.abs( y_true - y_pred ), np.maximum(y_true, y_pred) )
    
    # Mean inverse balanced relative error
    def mibre(self, y_true, y_pred):
        return np.mean(self.ibre(y_true, y_pred))
    
    # Spearman rank correlation coefficient
    # Only retorns coefficient, not p_value
    def spearmancc(self, y_true, y_pred):
        corr, p_value = spearmanr(y_true, y_pred)
        return corr
    