from enum import Enum, auto
import pandas as pd
from math import ceil
from utils import ps

class Missing_Value_Handling(Enum):
    NO = auto()
    ZERO = auto()
    MEAN = auto()
    MEDIAN = auto()


class Preprocessing:
    
    def __init__(self, remove_missing_columns = 1, remove_missing_rows = 1, missing_value_handling = Missing_Value_Handling.NO):
        self.remove_missing_columns = remove_missing_columns # Percentage
        self.remove_missing_rows = remove_missing_rows
        self.missing_value_handling = missing_value_handling
    
    def process(self, df):
        # remove columns with missing values
        if self.remove_missing_columns <= 1:
            thresh = int( ceil(df.shape[0] * self.remove_missing_columns) )
            df = df.dropna(axis = "columns", thresh = thresh)
        
        # Bugged
        # TODO fix
        # remove rows with missing values
#        if self.remove_missing_rows <= 1:
#            thresh = int( ceil(df.shape[1] * self.remove_missing_rows) )
#            df = df.dropna(axis = "index", thresh = thresh)
        
        # Replace missing values
        if self.missing_value_handling != Missing_Value_Handling.NO:
            mval = 0
            if self.missing_value_handling == Missing_Value_Handling.ZERO:
                mval = 0 # redundant
            elif self.missing_value_handling == Missing_Value_Handling.MEAN:
                mval = df.mean()
            elif self.missing_value_handling == Missing_Value_Handling.MEDIAN:
                mval = df.median()
            df.fillna(mval, inplace=True)
        
        return df