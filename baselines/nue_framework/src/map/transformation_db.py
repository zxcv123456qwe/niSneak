from map import Database
from scipy.stats import boxcox
from numpy import log, mean, std
from sklearn.preprocessing import FunctionTransformer
import pandas as pd
import numpy as np
from transformation import DataTransformation, StandardScaling, MinMaxScaling, LogMinMaxScaling

transformation_db = Database(DataTransformation, {"none" : FunctionTransformer,
                                      "norm" : StandardScaling,
                                      "log" : FunctionTransformer,
                                      "minmax" : MinMaxScaling,
                                      "logminmax" : LogMinMaxScaling
                                      },
                                     {"log" : {"func":np.log1p,
                                               "inverse_func":np.expm1
                                               },
                                      })
