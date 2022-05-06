from evaluation import MetricScorer

class D2H(MetricScorer):
    """
    Class:
        D2H
    Description:
        There is a point called heaven, which is the optimal place for an estimator.
        This metric measures the distance to that place.
        Multi-objective, requires the composite argument in self.
        Supports any metric.
    """

    def setConstants(self):
        self.name = "d2h"
        self.problem = "both"
        self.greater_is_better = False
        self.lo = 0
        self.hi = 1
        self.baseline = None
        self.unifeature = False
        self.composite = []

    def _score_func(self, y_true, y_pred, X, estimator):
        metrics = self.composite # Get composite metrics
        if len(metrics) == 0:
            return None
        distance = 0
        for m in metrics:
            hi = m.hi
            lo = m.lo
            x = m.evaluate(y_true, y_pred, X = X, estimator = estimator) # We currently only support normal metrics
            score = (x - lo) / (hi - lo) # Calculate normalized score
            if not(m.greater_is_better): # Account for loss metrics
                score = 1 - score
            score = max(0, score) # Bind in [0, 1] interval
            score = min(1, score)
            distance += (1 - score)**2
        distance = (distance / len(metrics))**(1/2)
        return distance
        