from utils import ps

class Learner(ps):
    """
    Class:
        Learner
    Description:
        Represents an scikit-learn object.
        Interface adds hyper-parameters for easiness at tuning.
        Supports classification/regression.
    Attributes:
        - name,str: Name of the learner, used only for searching purposes.
        - problem,str: Whether learner is for classification, regression, or both.
        - classification,class: scikit learn class used for classification, if any.
        - regression,class: scikit learn class used for regression, if any.
        - parameters,dict: default hyperparameters for either class.
    """
    def __init__(self, name, classification = None, regression = None, parameters = {}):
        """
        Function:
            __init__
        Description:
            Instances a Learner, storing name, class, and default parameters.
        Input:
            - name,str: name of the learner, used only for searching purposes.
            - classification,class: scikit learn class used for classification, if any.
            - regression,class: scikit learn class used for regression, if any.
            - parameters,dict: default hyperparameters for either class.
        Output:
            Instance of the Learner.
        """
        self.name = name
        self.classification = classification
        self.regression = regression
        self.parameters = parameters

        # Set the type of problem that this learner supports
        problem = "none"
        if (self.regression != None) and (self.classification != None):
            problem = "both"
        elif (self.regression != None):
            problem = "regression"
        elif (self.classification != None):
            problem = "classification"
        self.problem = problem
    
    def get_class(self, problem = None):
        """
        Function:
            get_class
        Description:
            Returns class reference of learner, depending on type of problem.
            If the type of problem is not specified, then we return what we support.
        Input:
            - problem,str: Type of problem, should be either "classification" or "regression".
        Output:
            If problem is specified:
                - If the problem is supported, returns the appropiate instance.
                - If problem is not supported, returns None.
            If problem is not specified:
                - If Learner supports only one problem, return that class.
                - If it supports both, returns dictionary:
                    {
                        "regression" : RegClass,
                        "classification" : ClaClass
                    }
                - If neither is supported, returns None.
        """
        # If we are asked to return one type of learner, we do
        # Doesnt matter if its not supported
        if problem == "regression":
            return self.regression
        if problem == "classification":
            return self.classification
        
        # If neither, and it is not none, it is undefined
        if problem is not None:
            return None

        # If parameter is none, we have to figure out what can we do
        if problem is None:
            if self.problem == "both":
                return { "regression" : self.regression, \
                        "classification" : self.classification }
            elif self.problem == "regression":
                return self.regression
            elif self.problem == "classification":
                return self.classification
        
        # If nothing fits, we return nothing (redundant)
        return None

