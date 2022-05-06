from modeling.base import BaseSPLModel
from modeling.pom3.pom3 import pom3
import random

class POM3Model( BaseSPLModel ):
    
    obj_names = ("cost", "score", "idle_rate" )
    obj_weights = (-1, 1, -1)
    
    def __init__(self, 
            culture = 1,
            criticality = 1,
            criticality_modifier = 1,
            initial_known = 1,
            inter_dependency = 1,
            dynamism = 1,
            size = 1,
            plan = 1, 
            team_size = 1
        ):
        self.culture = culture
        self.criticality = criticality
        self.criticality_modifier = criticality_modifier
        self.initial_known = initial_known
        self.inter_dependency = inter_dependency
        self.dynamism = dynamism
        self.size = size
        self.plan = plan 
        self.team_size = team_size

    def score(self):
        params = self.get_params()
        ind = [ params[key] for key in self._get_param_names() ]
        
        pom = pom3()
        output = pom.simulate(ind)
        return dict( [ (m, v) for m, v in zip( self.obj_names, output) ] )

if __name__ == "__main__":
    pom3a_bounds = {
            "culture" : (0.1, 0.9),
            "criticality" : (0.82, 1.20),
            "criticality_modifier" : (2, 10),
            "initial_known" : (0.40, 0.70),
            "inter_dependency" : (1, 100),
            "dynamism" : (1, 50),
            "size" : (0, 4),
            "plan" : (0, 5), 
            "team_size" : (1, 44)
        }
    
    sample = dict( [ (k, random.randint( v1, v2 ) 
                      if type(v1) == int
                      else random.uniform( v1, v2 ) ) for k, (v1, v2) in pom3a_bounds.items() ] )
    
    print(sample)
    
    x = POM3Model( 
                  **sample
                )
    res = x.score()
    print(res)
