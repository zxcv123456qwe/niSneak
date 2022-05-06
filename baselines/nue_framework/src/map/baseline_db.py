from baseline import MARP0
from baseline import MDARP0
from baseline import Median
from baseline import MARP0LOO

baseline_db = { "None" : None,
         "marp0" : MARP0(),
         "mdarp0" : MDARP0(),
         "median" : Median(),
         "marp0loo" : MARP0LOO()
         }
