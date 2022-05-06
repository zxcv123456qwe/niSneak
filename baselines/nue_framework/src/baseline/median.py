from baseline import Baseline
import numpy as np
from utils import ps

class Median(Baseline):
    """
    Class:
        Median
    Description:
        Deterministic prediction baseline algorithm.
        Always guesses using the median of actual data.
        Returns mean and standard deviation of predictions.
    """
    
    def predict(self, actual):
        res = 0
        std = 0
        n = actual.size
        actual = np.array(actual)
        
        prediction = np.median(actual)
        samples = np.abs( actual - prediction )
        
        res = np.mean(samples)
        std = np.std(samples)
            
        return ps(**{"center":res, "scale":std})
    
    