from utils import ps

class Optimizer(ps):
    
    def __init__(self, name, pt_class, parameters):
        self.name = name
        self.pt_class = pt_class
        self.parameters = parameters