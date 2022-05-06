from reading import Dataset
from map import Database

#dataset_isbsg = Dataset("1 - isbsg10", "csv", predict = "N_effort", exclude = ["N_effort_level1", "S_effort"], na_values = "?", as_dummies = True)
class Mirror:
    __getitem__ = lambda self, x : x

dataset_db = Database(Dataset, Mirror())
