# Metric implementation based on the xFair:
# https://github.com/anonymous12138/biasmitigation/blob/main/Measure.py

from evaluation import MetricScorer
from sklearn.metrics import confusion_matrix
import numpy as np

def flip_privilege(X, feature):
    """
    Function:
        flip_privilege
    Description:
        Flips values of privileged attribute.
        Attributes in unpriviledged remain the same.
        In other words, sets all values to 0
    Input:
        - X,dataframe: Columns that the model predicted on.
        - feature,str: Name of the feature.
    Output:
        Copy of dataframe
    """
    X_new = X.copy()
    X_new[feature] = np.where(X_new[feature]==1, 0, 1)
    return X_new

def slice_privilege(X):
    """
    Function:
        slice_privilege
    Description:
        Returns 2 masks, i.e. lists of booleans.
        Indicating if index corresponds with (un)privileged class.
        We assume privilege classes are 1 in the data.
    Input:
        - X,column: Protected attribute column.
    Output:
        2 lists of booleans.
        First is mask of privileged class.
        Second is mask for unprivileged class.
    """
    return X == 1, X == 0

def get_confusion(y_true, y_pred, X):
    """
    Function:
        get_confusion
    Description:
        Calculates confusion matrix of (un)priviledged class.
    Input:
        - y_true,list: List of actual y values.
        - y_pred,list: List of predicted y values.
        - X,dataframe: Columns that the model predicted on.
        - feature,str: Name of the feature.
    Output:
        2 tiered dictionary with this structure:
        {
            "priv" : {
                "tp" : tp,
                "tn" : tn,
                "fp" : fp,
                "fn" : fn
            },
            "unpr" : {
                "tp" : tp,
                "tn" : tn,
                "fp" : fp,
                "fn" : fn
            },
        }
    """
    priv, unpriv = slice_privilege(X)
    priv_true, priv_pred = y_true[priv], y_pred[priv]
    unpr_true, unpr_pred = y_true[unpriv], y_pred[unpriv]
    
    res = {}
    for name, true, pred in zip( ["priv", "unpr"], [priv_true, unpr_true], [priv_pred, unpr_pred] ):
        result = confusion_matrix(true, pred).ravel()
        tn, fp, fn, tp = 0, 0, 0, 0
        if len(result) == 1:
            if  np.array_equal(y_true, y_pred):
                tp = result[0]
            else:
                tn = result[0]
        elif len(result) == 4:
            tn, fp, fn, tp = confusion_matrix(true, pred).ravel()
        res[name] = {
            "tp" : tp,
            "tn" : tn,
            "fp" : fp,
            "fn" : fn
        }
    return res
    

class AOD(MetricScorer):
    """
    Class:
        AOD
    Description:
        MetricX of Average Odds Difference
        Average of difference in False Positive Rates(FPR)
        and True Positive Rates(TPR) for unprivileged and privileged groups.
    """
    
    def setConstants(self):
        self.name = "aod"
        self.problem = "classification"
        self.greater_is_better = False
        self.lo = 0
        self.hi = 1
        self.unifeature = True
        self.baseline = None
    
    def _score_func(self, y_true, y_pred, X, estimator):
        X = X[ self.feature ]
        conf = get_confusion(y_true, y_pred, X)
        if (conf["unpr"]["tp"] + conf["unpr"]["fn"] > 0) and (conf["priv"]["tp"] + conf["priv"]["fn"] > 0)\
            and ( conf["unpr"]["fp"] + conf["unpr"]["tn"] > 0) and ( conf["priv"]["fp"] + conf["priv"]["tn"] > 0 ):
            tpr_unpr = conf["unpr"]["tp"]/( conf["unpr"]["tp"] + conf["unpr"]["fn"] )
            fpr_unpr = conf["unpr"]["fp"]/( conf["unpr"]["fp"] + conf["unpr"]["tn"] )
            tpr_priv = conf["priv"]["tp"]/( conf["priv"]["tp"] + conf["priv"]["fn"] )
            fpr_priv = conf["priv"]["fp"]/( conf["priv"]["fp"] + conf["priv"]["tn"] )
            return np.abs(((fpr_unpr - fpr_priv) + (tpr_unpr - tpr_priv)) / 2)
        else:
            # If there are no pos/neg predictions for (un)protected
            # Thats very biased
            return 1


class EOD(MetricScorer):
    """
    Class:
        EOD
    Description:
        MetricX of Equal Opportunity Difference
        Difference of True Positive Rates(TPR) for unprivileged and privileged groups.
    """
    
    def setConstants(self):
        self.name = "eod"
        self.problem = "classification"
        self.greater_is_better = False
        self.lo = 0
        self.hi = 1
        self.unifeature = True
        self.baseline = None
    
    def _score_func(self, y_true, y_pred, X, estimator):
        X = X[ self.feature ]
        conf = get_confusion(y_true, y_pred, X)
        if conf["unpr"]["tp"] + conf["unpr"]["fn"] > 0 and conf["priv"]["tp"] + conf["priv"]["fn"] > 0:
            tpr_unpr = conf["unpr"]["tp"]/( conf["unpr"]["tp"] + conf["unpr"]["fn"] )
            tpr_priv = conf["priv"]["tp"]/( conf["priv"]["tp"] + conf["priv"]["fn"] )
            return np.abs(tpr_unpr - tpr_priv)
        else:
            # No TP and FN?
            # Thats very biased
            return 1
        

class SPD(MetricScorer):
    """
    Class:
        SPD
    Description:
        MetricX of Statistical Party Difference
        Difference between probability of unprivileged group
        (protected attribute PA = 0) gets favorable prediction (Y = 1)
        & probability of privileged group (protected attribute PA = 1)
        gets favorable prediction (Y = 1)
    """
    
    def setConstants(self):
        self.name = "spd"
        self.problem = "classification"
        self.greater_is_better = False
        self.lo = 0
        self.hi = 1
        self.unifeature = True
        self.baseline = None
    
    def _score_func(self, y_true, y_pred, X, estimator):
        X = X[ self.feature ]
        conf = get_confusion(y_true, y_pred, X)
        if conf["priv"]["tp"] + conf["priv"]["fp"] + conf["priv"]["tn"] + conf["priv"]["fn"] > 0 \
                and conf["unpr"]["tp"] + conf["unpr"]["fp"] + conf["unpr"]["tn"] + conf["unpr"]["fn"] > 0:
            p_favo_priv = (conf["priv"]["tp"] + conf["priv"]["fp"]) / (conf["priv"]["tp"] + conf["priv"]["fp"] + conf["priv"]["tn"] + conf["priv"]["fn"])
            p_favo_unpr = (conf["unpr"]["tp"] + conf["unpr"]["fp"]) / (conf["unpr"]["tp"] + conf["unpr"]["fp"] + conf["unpr"]["tn"] + conf["unpr"]["fn"])
            return np.abs(p_favo_unpr - p_favo_priv)
        else:
            return 1

class DI(MetricScorer):
    """
    Class:
        DI
    Description:
        MetricX of Disparate Impact
        Ratio between probability of unprivileged group
        (protected attribute PA = 0) gets favorable prediction (Y = 1)
        & probability of privileged group (protected attribute PA = 1)
        gets favorable prediction (Y = 1)
    """
    
    def setConstants(self):
        self.name = "di"
        self.problem = "classification"
        self.greater_is_better = False
        self.lo = 0
        self.hi = 1
        self.unifeature = True
        self.baseline = None
    
    def _score_func(self, y_true, y_pred, X, estimator):
        X = X[ self.feature ]
        conf = get_confusion(y_true, y_pred, X)
        if conf["priv"]["tp"] + conf["priv"]["fp"] + conf["priv"]["tn"] + conf["priv"]["fn"] > 0 \
                and conf["unpr"]["tp"] + conf["unpr"]["fp"] + conf["unpr"]["tn"] + conf["unpr"]["fn"] > 0:
            p_favo_priv = (conf["priv"]["tp"] + conf["priv"]["fp"]) / (conf["priv"]["tp"] + conf["priv"]["fp"] + conf["priv"]["tn"] + conf["priv"]["fn"])
            p_favo_unpr = (conf["unpr"]["tp"] + conf["unpr"]["fp"]) / (conf["unpr"]["tp"] + conf["unpr"]["fp"] + conf["unpr"]["tn"] + conf["unpr"]["fn"])
            if p_favo_priv != 0:
                return np.abs(1 - p_favo_unpr / p_favo_priv)
            else:
                # Biased against 'priviledged' class
                return 1
        else:
            return 1

class FR(MetricScorer):
    """
    Class:
        FR
    Description:
        MetricFull of Flip Rate
        The ratio of instances whose predicted label (Y) will change
        when flipping their protected attributes (e.g., PA=1 to PA=0).
    """

    def setConstants(self):
        self.name = "fr"
        self.problem = "classification"
        self.greater_is_better = False
        self.lo = 0
        self.hi = 1
        self.unifeature = True
        self.baseline = None

    def _score_func(self, y_true, y_pred, X, estimator):
        X_flip = flip_privilege(X, self.feature)
        y_flip = np.array( estimator.predict(X_flip) )
        n = X.shape[0]
        same = np.count_nonzero( np.array(y_pred) == y_flip )
        return (n-same)/n
        