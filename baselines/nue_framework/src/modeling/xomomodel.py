from modeling.base import BaseSPLModel
from modeling.xomo.xomo_liaison import xomol
import random

class XOMOModel( BaseSPLModel ):
    
    obj_names = ( "effort", "months", "defects", "risks" )
    obj_weights = (-1, -1, -1, -1)
    obj_hi = (20000, 200, 50000, 15)
    obj_lo = (0,0,0,0)
    
    def __init__(self, 
            aa = 1,
            sced = 1.00,
            cplx = 0.73,
            site = 0.80,
            resl = 1.41, 
            acap = 0.71, 
            etat = 1,
            rely = 0.82,
            Data = 0.90, 
            prec = 1.24,
            pmat = 1.56,
            aexp = 0.81,
            flex = 1.01,
            pcon = 0.81,
            tool = 0.78,
            time = 1.00,
            stor = 1.00,
            docu = 0.81,
            b = 3,
            plex = 0.85,
            pcap = 0.76,
            kloc = 2,
            ltex = 0.84, 
            pr = 1,
            ruse = 0.95,
            team = 1.01,
            pvol = 0.87):
        self.aa = aa
        self.sced = sced
        self.cplx = cplx
        self.site = site
        self.resl = resl
        self.acap = acap
        self.etat = etat
        self.rely = rely
        self.Data = Data
        self.prec = prec
        self.pmat = pmat
        self.aexp = aexp
        self.flex = flex
        self.pcon = pcon
        self.tool = tool
        self.time = time
        self.stor = stor
        self.docu = docu
        self.b = b
        self.plex = plex
        self.pcap = pcap
        self.kloc = kloc
        self.ltex = ltex
        self.pr = pr
        self.ruse = ruse
        self.team = team
        self.pvol = pvol

    def score(self):
        params = self.get_params()
        ind = [ params[key] for key in self._get_param_names() ]
        
        xomoxo = xomol()
        output = xomoxo.run(ind)
        return dict( [ (m, v) for m, v in zip( self.obj_names, output) ] )

if __name__ == "__main__":
    common_bounds = {
            "aa": (1, 6),
            "sced": (1.00, 1.43),
            "cplx": (0.73, 1.74),
            "site": (0.80, 1.22),
            "resl": (1.41, 7.07),
            "acap": (0.71, 1.42),
            "etat": (1, 6),
            "rely": (0.82, 1.26),
            "Data": (0.90, 1.28),
            "prec": (1.24, 6.20),
            "pmat": (1.56, 7.80),
            "aexp": (0.81, 1.22),
            "flex": (1.01, 5.07),
            "pcon": (0.81, 1.29),
            "tool": (0.78, 1.17),
            "time": (1.00, 1.63),
            "stor": (1.00, 1.46),
            "docu": (0.81, 1.23),
            "b": (3, 10),
            "plex": (0.85, 1.19),
            "pcap": (0.76, 1.34),
            "kloc": (2, 1000),
            "ltex": (0.84, 1.20),
            "pr": (1, 6),
            "ruse": (0.95, 1.24),
            "team": (1.01, 5.48),
            "pvol": (0.87, 1.30)
        }
    
    sample = dict( [ (k, random.randint( v1, v2 ) 
                      if type(v1) == int
                      else random.uniform( v1, v2 ) ) for k, (v1, v2) in common_bounds.items() ] )
    
    print(sample)
    
    x = XOMOModel( 
                  **sample
                )
    res = x.score()
    print(res)
