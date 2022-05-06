class ps:
    """
    Class:
        ps
    Description:
        Printable and Subscriptable object
        Based on class o by timm
        https://github.com/txt/sin21/blob/main/docs/hw5.md
    """
    def __init__(self, **kwargs):
        """
            Function:
                __init__
            Description:
                Initializes the pr object
            Input:
                - Any class arguments
            Output:
                pr object
        """
        self.__dict__.update(kwargs)
    
    def __repr__(self):
        """
            Function:
                __repr__
            Description:
                Returns representation of the object, as str.
            Input:
                None
            Output:
                str representation
        """
        return "{"+ ', '.join([f"{k}:{v}" for k, v in self.__dict__.items() if  k[0] != "_"]) + "}"
    
    def __getitem__(self, key):
        """
            Function:
                __getitem__
            Description:
                Returns attribute of the class.
            Input:
                -key, str: attribute name
            Output:
                Corresponding attribute value to key.
                If undefined, returns None.
        """
        return self.__dict__.get(key)
