from baseline import Baseline
import numpy as np
from utils import ps

class MARP0LOO(Baseline):
    """
    Class:
        MARP0LOO
    Description:
        Random prediction baseline algorithm.
        Randomly guesses by picking one of the actual values (except actual result).
        Returns mean and standard deviation of predictions.
        Especial for using leave-one-out cross-validation.
    """
    
    def predict(self, actual):
        res = 0
        std = 0
        n = actual.size
        actual = np.array(actual)
        samples = []
        
        # Original SA
        # We run it to calculate std, regardless of method
        for i in range(0, n):
            p = [1/(actual.size) for x in range(actual.size)]
            pred = np.random.choice(actual, self.n_runs, replace=True, p=p)
            samples.extend( np.abs( pred - actual[i] ) )
        res = np.mean(samples)
        
        # Depending on size of sample
        # Use original MARP0 by Shepperd and MacDonell
        # Or use unbiased version by Langdon et al.
        if n <= 2000:
            res = 0
            # Unbiased
            for i in range(0, n):
                for j in range(0, i):
                    res += abs( actual[i] - actual[j] )
            res *= 2.0 / ( n ** 2 )
           
        
        std = np.std(samples)
            
        return ps(**{"center":res, "scale":std})
    
    