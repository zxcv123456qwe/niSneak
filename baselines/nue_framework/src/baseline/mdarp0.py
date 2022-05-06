from baseline import Baseline
import numpy as np
from utils import ps

class MDARP0(Baseline):
    """
    Class:
        MdARP0
    Description:
        Random prediction baseline algorithm.
        Randomly guesses by picking one of the actual values (except actual result).
        Returns median and standard deviation instead of mean.
    """
    
    def predict(self, actual):
        res = 0
        std = 0
        n = actual.size
        actual = np.array(actual)
        
        samples = []
        for i in range(0, n):
            p = [0 if x == i else 1/(actual.size - 1) for x in range(actual.size)]
            pred = np.random.choice(actual, self.n_runs, replace=True, p=p)
            samples.extend( np.abs( actual[i] - pred))
            
        res = np.median(samples)
        std = np.std(samples)
            
        return ps(**{"center":res, "scale":std})
    
    