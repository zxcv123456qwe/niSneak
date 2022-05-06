from evaluation import MetricScorer
from sklearn.metrics import accuracy_score, precision_score,\
    recall_score, f1_score, confusion_matrix
import numpy as np

def false_negative_score( y_true, y_pred ):
    result = confusion_matrix(y_true, y_pred).ravel()
    tn, fp, fn, tp = 0, 0, 0, 0
    if len(result) == 1:
        if  np.array_equal(y_true, y_pred):
            tp = result[0]
        else:
            tn = result[0]
    elif len(result) == 4:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    if (fp == 0 and tn == 0):
        return 0
    return fp/( fp + tn )

class Accuracy(MetricScorer):
    
    def setConstants(self):
        self.name = "accuracy"
        self.problem = "classification"
        self.greater_is_better = True
        self.lo = 0
        self.hi = 1
        self.baseline = None
        self.unifeature = False
        self.composite = None
    
    def _score_func(self, y_true, y_pred, X, estimator):
        return accuracy_score( y_true, y_pred )
    

class Precision(MetricScorer):
    
    def setConstants(self):
        self.name = "precision"
        self.problem = "classification"
        self.greater_is_better = True
        self.lo = 0
        self.hi = 1
        self.baseline = None
        self.unifeature = False
        self.composite = None
        self.zero_division = 0
    
    def _score_func(self, y_true, y_pred, X, estimator):
        return precision_score( y_true, y_pred, zero_division = self.zero_division )

class Recall(MetricScorer):
    
    def setConstants(self):
        self.name = "recall"
        self.problem = "classification"
        self.greater_is_better = True
        self.lo = 0
        self.hi = 1
        self.baseline = None
        self.unifeature = False
        self.composite = None
        self.zero_division = 0
    
    def _score_func(self, y_true, y_pred, X, estimator):
        return recall_score( y_true, y_pred, zero_division = self.zero_division )

class F1(MetricScorer):
    
    def setConstants(self):
        self.name = "f1"
        self.problem = "classification"
        self.greater_is_better = True
        self.lo = 0
        self.hi = 1
        self.baseline = None
        self.unifeature = False
        self.composite = None
        self.zero_division = 0
    
    def _score_func(self, y_true, y_pred, X, estimator):
        return f1_score( y_true, y_pred, zero_division = self.zero_division )

class FalseAlarm(MetricScorer):
    
    def setConstants(self):
        self.name = "falsealarm"
        self.problem = "classification"
        self.greater_is_better = False
        self.lo = 0
        self.hi = 1
        self.baseline = None
        self.unifeature = False
        self.composite = None
        self.zero_division = 0
    
    def _score_func(self, y_true, y_pred, X, estimator):
        return false_negative_score( y_true, y_pred )
