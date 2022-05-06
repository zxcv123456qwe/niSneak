from abc import ABC, abstractmethod

class Baseline(ABC):
    """
    Class:
        Baseline
    Description:
        Abstract class.
        Baseline prediction algorithms.
        Very simple, used for the calculation of metrics.
    Attributes:
        - n_runs,int: If baseline is stochastic, amount of repetitions
    """
    def __init__(self, n_runs = 1000):
        """
        Function:
            __init__
        Description:
            Instances a Baseline.
        Input:
            - n_runs,int: If baseline is stochastic, amount of repetitions
        Output:
            Instance of the Baseline.
        """
        self.n_runs = n_runs
    
    @abstractmethod
    def predict(self, actual):
        """
        Function:
            predict
        Description:
            Returns the distribution of the prediction.
            It is a ps object with 2 keys/attributes: center, and scale
            The value of center is a representative (mean, median) of the baseline
            The value of scale is a measure of spread (sd, iqr) of the baseline
        Input:
            - actual,column: The list of real values, as a pandas column
        Output:
            A ps object with the structure:
            {
                "center" : center_value,
                "scale" : scale_value
            }
        """
        pass
    