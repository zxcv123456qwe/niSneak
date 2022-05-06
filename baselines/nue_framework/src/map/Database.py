class Database():
    """
    Class:
        Database
    Description:
        Storage for parts of the machine learning process.
        Each database represents one technique type.
        e.x.: Data transformations, parameter optimizers.
        Used to convert names into appropriate classes.
        Values are returned as container class "classtype".
    Attributes:
        - classtype,class: Container class, all returned items are of this type.
        - mapping,dict: mapping of name into one class.
        - base_params: mapping of name into default parameters.
    """
    def __init__(self, classtype, mapping = {}, base_params = {}):
        """
        Function:
            __init__
        Description:
            Returns an instance of Database.
        Input:
            - classtype,class: Container class, all returned items are of this type.
            - mapping,dict: mapping of name into one class
            - base_params: mapping of name into default parameters
        Output:
            Instance of DatabaseNoClass
        """
        self.classtype = classtype
        self.mapping = mapping
        self.base_params = base_params
    
    def get(self, items):
        """
        Function:
            get
        Description:
            Returns a list of container classes, if applicable.
            If items are not matched, returns whatever it can.
        Input:
            - items,list: Pairs of (name, params) values
        Output:
            List of instances of container class.
            Size is at most equal to input list.
            If a certain name is not found, nothing is included.
        """
        res = []
        for item in items:
            x = self.get_one(item)
            if x is not None:
                res.append(x)
        return res
    
    def get_one(self, item):
        """
        Function:
            get_one
        Description:
            Gets container class and hyper params.
            From a name and (user defined) hyper parameters of a method.
            User-defined parameters are given priority.
        Input:
            - items,list: Pairs of (name, params) values
        Output:
            List with instances of the container class
        """
        try:
            name, params = item
            if name.lower() in self.base_params.keys():
                base = self.base_params[name].copy()
            else:
                base = {}
            for k in params:
                base[k] = params[k]
            return self.classtype( name, self.mapping[name.lower()], base )
        except:
            return None


class DatabaseTwoClass(Database):
    """
    Class:
        DatabaseTwoClass
    Description:
        Extends Database to store objects that have two possible classes.
        e.x. Learners (classification and regression).
    Attributes:
        Same as Database
    """

    def get_one(self, item):
        """
        Function:
            get_one
        Description:
            Gets container class and hyper params.
            From a name and (user defined) hyper parameters of a method.
            User-defined parameters are given priority
        Input:
            - item,list: Pair of (name, params) values
        Output:
            Instance of the container class.
        """
        try:
            name, params = item
            if name.lower() in self.base_params.keys():
                base = self.base_params[name].copy()
            else:
                base = {}
            for k in params:
                base[k] = params[k]
            return self.classtype( name, self.mapping[name.lower()][0],\
                    self.mapping[name.lower()][1], base )
        except:
            return None

class DatabaseNoClass(Database):
    """
    Class:
        DatabaseNoClass
    Description:
        Extends Database to store objects that have no classes.
        e.x. Metrics, which are just a formula.
    Attributes:
        Same as Database, with the exception of mapping.
        Class is also contained on each instance, under name "class"
    """

    def __init__(self, base_params = {}):
        """
        Function:
            __init__
        Description:
            Returns an instance of DatabaseNoClass.
        Input:
            - base_params: mapping of name into default parameters, including class
        Output:
            Instance of DatabaseNoClass
        """
        self.base_params = base_params

    def get_one(self, item):
        """
        Function:
            get_one
        Description:
            Gets container class.
            From a name and (user defined) settings.
            User-defined settings are given priority
        Input:
            - item,list: Pair of (name, settings) values
        Output:
            Instance of the container class.
        """
        try:
            name, params = item
            if name.lower() in self.base_params.keys():
                base = self.base_params[name].copy()
                base.pop("class")
            else:
                base = {}
            for k in params:
                base[k] = params[k]
            # Check name override
            real_name = name
            if "name" in base.keys():
                name = base["name"]
                base.pop("name")
            return self.base_params[real_name]["class"]( name, **base )
        except Exception as e:
            raise e
            return None

