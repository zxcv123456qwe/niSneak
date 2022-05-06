from utils import ps

class Model(ps):
    
    def __init__(self, name, mo_class, parameters):
        self.name = name
        self.mo_class = mo_class
        self.parameters = parameters