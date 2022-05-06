def distance_to_heaven(self, y_true, y_pred, **kwargs):
    """
        There is a point called heaven, which is the optimal place for an estimator.
        This metric measures the distance to that place.
        Multi-objective, requires the composite argument in self.
    """
    metrics = self.composite # Get composite metrics
    distance = 0
    for m in metrics:
        hi = m.hi
        lo = m.lo
        x = m.evaluate(y_true, y_pred) # We currently only support normal metrics
        score = (hi - x) / (hi - lo) # Calculate normalized score
        if m.greater_is_better: # Account for loss metrics
            score = 1 - score
        score = max(0, score) # Bind in [0, 1] interval
        score = min(1, score)
        distance += (1 - score)**2
    distance = distance**(1/2)
    return distance
