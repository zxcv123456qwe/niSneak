from utils import ps

class AttributeSelector(ps):
    
    def __init__(self, name, as_class, parameters):
        self.name = name
        self.as_class = as_class
        self.parameters = parameters
    