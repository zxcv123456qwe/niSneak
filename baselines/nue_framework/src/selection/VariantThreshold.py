from sklearn.feature_selection import VarianceThreshold
import numpy as np

class VariantThreshold(VarianceThreshold):
    
    def fit(self, X, y=None):
        try:
            return super().fit(X, y)
        except ValueError:
            # No feature meets the threshold
            # Return only the feature with highest variance
            self.variances_[np.argmax(self.variances_)] = self.threshold + 1
            return self
            
