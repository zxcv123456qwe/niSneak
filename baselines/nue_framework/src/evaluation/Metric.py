from utils import ps, get_problem_type
from sklearn.metrics import make_scorer
from inspect import signature
from abc import ABC, abstractmethod
from sklearn.metrics._scorer import _PredictScorer

# Local baseline DB
# To avoid circular imports
from baseline import MARP0
from baseline import MDARP0
from baseline import Median
from baseline import MARP0LOO

baselines = { "None" : None,
         "marp0" : MARP0(),
         "mdarp0" : MDARP0(),
         "median" : Median(),
         "marp0loo" : MARP0LOO()
}

class MetricScorer(ABC, ps, _PredictScorer):
    """
    Class:
        MetricScorer
    Description:
        Abstract class, requires definition of score method
        Represents an evaluation metric that requires the complete dataset
        as well as the prediction method.
        Focuses on one feature at a time.
    Attributes:
        - feature,str: name of feature to calculate metric.
    Constants: Should be set by constructor of each subclass
        - name,str: Name of metric
        - problem,str: Whether metric is for classification, regression, or both.
        - greater_is_better,bool: Whether a learner/optimizer should increase this metric or not.
        - lo,float or None: Reference point, theorethical lowest possible value.
        - hi,float or None: Reference point, theorethical highest possible value.
        - baseline,class: Baseline object to calculate this metric.
    """
    
    def __init__(self, name = None, feature = None, **kwargs):
        """
        Function:
            __init__
        Description:
            Instances a MetricX, storing all attributes.
        Input:
            - name,str: Unused
            - feature,str: name of feature to calculate metric.
        Output:
            Instance of the MetricX.
        """
        self.feature = feature
        self.setConstants()
        # Override name
        if name is not None:
            self.name = name
        if feature != None:
            self.name += "-" + feature
        self._sign = 1 if self.greater_is_better else -1
        self._kwargs = {}
        for k, v in kwargs.items():
            setattr(self, k, v)
        if self.baseline is not None:
            self.baseline = self.baseline() # New instance
    
    @abstractmethod
    def setConstants(self):
        """
        Function:
            setConstants
        Description:
            Sets the constants of the MetriX.
            Should be implemented by each subclass.
            This is a template.
        Input:
            None.
        Output:
            None. Should modify attributes.
        """
        self.name = ""
        self.problem = None
        self.greater_is_better = None
        self.lo = None
        self.hi = None
        self.baseline = None
        self.composite = None
    
    def _score(self, method_caller, estimator, X, y_true, sample_weight=None):
        """
        Function:
            _score
        Description:
            Scikit learn score.
            We override it to be able to access X and estimator as we calculate the metric.
            Description from scikit-learn:
         
        Evaluate predicted target values for X relative to y_true.
        Parameters
        ----------
        method_caller : callable
            Returns predictions given an estimator, method name, and other
            arguments, potentially caching results.
        estimator : object
            Trained estimator to use for scoring. Must have a `predict`
            method; the output of that is used to compute the score.
        X : {array-like, sparse matrix}
            Test data that will be fed to estimator.predict.
        y_true : array-like
            Gold standard target values for X.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.
        Returns
        -------
        score : float
            Score function applied to prediction of estimator on X.
        """

        y_pred = method_caller(estimator, "predict", X)
        if sample_weight is not None:
            return self._sign * self._score_func(
                y_true, y_pred, X, estimator, sample_weight=sample_weight, **self._kwargs
            )
        else:
            return self._sign * self._score_func(y_true, y_pred, X, estimator, **self._kwargs)
    
    def make_scorer(self):
        return self
    
    @abstractmethod
    def _score_func(self, y_true, y_pred, X = None, estimator = None):
        """
        Function:
            _score_func
        Description:
            Calculate and return the metric.
        Input:
            - y_true,list: List of actual y values.
            - y_pred,list: List of predicted y values.
            - X,dataframe: Columns that the model predicted on.
        Output:
            None. Should modify attributes.
        """
        pass
    
    def get_formula(self):
        return self._score_func
    
    def evaluate(self, y, y_pred, X = None, estimator = None):
        return self._score_func(y, y_pred, X, estimator)

class GenericMetric:
    
    def __init__(self, name, sign, lo, hi):
        self.name = name
        self._sign = sign
        self.greater_is_better = True if sign == 1 else False
        self.lo = lo
        self.hi = hi
