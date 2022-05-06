# File with formulas for metrics
# For software effort estimation
# All formulas start with self,
# As they become attributes of a Metric object

import numpy as np
from scipy.stats.mstats import gmean
from scipy.stats import spearmanr
from baseline import MARP0
from baseline import MDARP0
from baseline import MARP0LOO

# Multi objective metric related functions
# def reference_point(self):
#     # If we get a list the same size of all metrics, return that
#     # If not, return -1 for all values
#     # Useful for standardized techniques; i.e. sa and sd
    
#     if len(self.reference) == len( self.multiobj ):
#         reference = self.reference
#     else:
#         reference = [-1 for i in range(len(self.multiobj))]
    
#     return dict( zip(self.multiobj, reference) )

# Similar to evaluate, but based on multiobj list
# def evaluate_multiobj(self, y_true, y_pred):
#     res = {}
#     for metric in self.multiobj:
#         r = self.get_function(metric)(y_true, y_pred)
#         res[metric] = r
#     return res

## Multi objective metrics

# Single-point hyper_volume
# Calculated as the product of all metrics
# def hyper_volume(self, y_true, y_pred):
#     scores = list(self.evaluate_multiobj(y_true, y_pred).values())
#     reference = list(self.reference_point().values())
#     diff = np.subtract(scores, reference)
    
#     # Dont accept negative values
#     # If any are negative, they are converted to 0
#     diff = [ max(0, v) for v in diff ]
    
#     return np.product( diff )


## Single objective metrics

def effect_size(self, y_true, y_pred):
    """
        Effect size, also known as Glass's delta.
        How different is a prediction with respect to baseline.
    """
    if self.baseline is not None:
        base_c = self.baseline.predict( y_true )["center"]
        base_s = self.baseline.predict( y_true )["scale"]
        if type(self.baseline) in [MARP0, MARP0LOO]:
            val = mar(self, y_true, y_pred)
        elif type(self.baseline) == MDARP0:
            val = mdar(self, y_true, y_pred)
        else: return None
        
        if base_s == 0:
            return np.abs( ( val - base_c + 1 ) / (base_s + 1) )
        else:
            return np.abs( ( val - base_c ) / base_s )
    return None

def sa(self, y_true, y_pred):
    """
        Standarized accuracy.
        Mean/Median absolute error of prediction.
        Standardized by a baseline estimator.
    """
    if self.baseline is not None:
        base = self.baseline.predict( y_true )["center"]
        if type(self.baseline) in [MARP0, MARP0LOO]:
            val = mar(self, y_true, y_pred)
        elif type(self.baseline) == MDARP0:
            val = mdar(self, y_true, y_pred)
        else: return None
        
        if base == 0:
            if val == 0:
                return 1
            return 1 - ( (val + 1) / (base+1) )
        else:
            return 1 - ( val / base )
    return None

def ae_i(self, y_true, y_pred):
    """
        Absolute error, aka absolute resuidual or AR.
    """
    return np.abs(y_true - y_pred)

def mre(self, y_true, y_pred):
    """
        Magnitude of relative error.
        Absolute error divided by the size of actual value.
    """
    mmre_l = []
    for y_t, y_p in zip(y_true, y_pred):
        num = np.abs(y_p - y_t)
        den = np.abs(y_t)
        if den == 0:
            if num != 0:
                den+=1
                num+=1
                mmre_l.append(num/den)
            else:
                mmre_l.append(0)
        else:
            mmre_l.append(num/den)
    return np.array(mmre_l)

def mmre(self, y_true, y_pred):
    """
        Mean magnitude of relative error.
    """
    return np.mean(mre(self, y_true, y_pred))

def mmre100(self, y_true, y_pred):
    """
        Mean magnitude of relative error, as percentage.
    """
    return mmre(self, y_true, y_pred) * 100

def mdmre(self, y_true, y_pred):
    """
        Median magnitude of relative error.
    """
    return np.median(mre(self, y_true, y_pred))

def mdmre100(self, y_true, y_pred):
    """
        Median magnitude of relative error, as percentage.
    """
    return self.mdmre(y_true, y_pred) * 100

def pred(self, n, y_true, y_pred):
    """
        PRED(X), usually PRED(25). % of predictions above X% of the MRE.
    """
    if y_true.size == 0:
        return 0
    return np.sum( mre(self, y_true, y_pred) <= (n/100) ) / y_true.size

def pred25(self, y_true, y_pred):
    """
        % of predictions below 25% of the MRE.
    """
    return pred(self, 25, y_true, y_pred)

def pred40(self, y_true, y_pred):
    """
        % of predictions below 40% of the MRE.
    """
    return pred(self, 40, y_true, y_pred)

def mar(self, y_true, y_pred):
    """
        Mean absolute residual, aka mean absolute error or mae.
    """
    return np.average(ae_i(self, y_true, y_pred))

def mdar(self, y_true, y_pred):
    """
        Median absolute residual, aka median absolute error or mdae or mdae.
    """
    return np.median(ae_i(self, y_true, y_pred))

def sdar(self, y_true, y_pred):
    """
        Standard deviation of absolute residual.
    """
    return np.std(ae_i(self, y_true, y_pred))

def sd(self, y_true, y_pred):
    """
        Standarized deviation.
        Based off the stability ratio.
        With respect to baseline.
    """
    if self.baseline is not None:
        base = self.baseline.predict( y_true )["scale"]
        if type(self.baseline) in [MARP0, MDARP0, MARP0LOO]:
            val = sdar(self, y_true, y_pred)
        else: return None
        
        if base == 0:
            return 1 - ((val + 1) / (base + 1))
        else:
            return 1 - (val / base)
    return None

def gmar(self, y_true, y_pred):
    """
        Geometric mean of the absolute residual.
    """
    return gmean(ae_i(self, y_true, y_pred))

def bre(self, y_true, y_pred):
    """
        Balanced relative error.
    """
    return np.divide(np.abs( y_true - y_pred ), np.minimum(y_true, y_pred) )

def mbre(self, y_true, y_pred):
    """
        Mean balanced relative error.
    """
    return np.mean(bre(self, y_true, y_pred))

def ibre(self, y_true, y_pred):
    """
        Inverse balanced relative error.
    """
    return np.divide(np.abs( y_true - y_pred ), np.maximum(y_true, y_pred) )

def mibre(self, y_true, y_pred):
    """
        Mean inverse balanced relative error.
    """
    return np.mean(ibre(self, y_true, y_pred))

def spearmancc(self, y_true, y_pred):
    """
        Spearman rank correlation coefficient.
        Only returns coefficient, not p_value.
    """
    corr, p_value = spearmanr(y_true, y_pred)
    return corr