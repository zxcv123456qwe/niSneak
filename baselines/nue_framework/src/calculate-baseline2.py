# Libraries
import numpy as np
import pandas as pd

# TODO fix imports
from database.dataset_db import dataset_db
from database.dt_db import dt_db
from database.as_db import as_db
from database.mlt_db import mlt_db
from database.pt_db import pt_db
from database.bl_db import bl_db
from database.cv_db import cv_db
from sklearn.metrics import make_scorer
from comp.Loader import Loader
from comp.LoaderCV import LoaderCV
from comp.Evaluation import Evaluation
from comp.Preprocessing import Preprocessing
from joblib import Memory
from shutil import rmtree
from comp.Selector import NumericalSelector
from comp.Imputer import FillImputer, SimplerImputer, KNNImputerDF
from comp.DataTransformation import OneHotEncoding


from sklearn.pipeline import Pipeline
from comp.Pipeline import FeatureJoin
import datetime
import time

# Ingnore convergence warnings
from  warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

# Prelude


# Data sets
#datasets = [db.dataset_isbsg]

# Load configuration)
FW, DS, DT, AS, PT, LA, EM = Loader().load_config()
datasets = dataset_db.get(DS)
data_transformation = dt_db.get(DT)
as_techniques = as_db.get(AS)
pt_techniques = pt_db.get(PT)
ml_techniques = mlt_db.get(LA)

# Pre processing
# prep = Preprocessing(remove_missing_columns = 0.25, missing_value_handling  = Missing_Value_Handling.MEAN)

prep = Preprocessing( **(FW["preprocessing"] if "preprocessing" in FW.keys() else {}) )

# Baseline
baseline = bl_db[ FW["baseline"][0] if "baseline" in FW.keys() else "none" ]

# Evaluation metrics
eva = Evaluation(EM, baseline)

# Cross Validation
# Defaults to 80:20 train test split
cv = cv_db[ FW["cv"][0] if "cv" in FW.keys() else "traintestsplit" ]( **(FW["cv"][1] if "cv" in FW.keys() else {"n_splits":1, "test_size":0.2}) )


# Output dataframe
out_cols = ["DS", "Iteration", "Center", "Spread"]
output_df = pd.DataFrame(columns=out_cols)
output_file = "baseline-" + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + ".csv"

# Dataset Loop
for ds in datasets:
    ds.set_datapath("data/")
    dataframe = ds.get_dataframe()
    
    # Data pre-processing
    # dataframe = prep.process(dataframe)
    X = dataframe[ dataframe.columns.difference([ds.predict]) ]
    Y = dataframe[ ds.predict ]
    
    if type(cv) == LoaderCV:
        cv.set_data( ds.id )
    
    # Cross validation
    for iteration, (train_index, test_index) in enumerate(cv.split(dataframe)):        
        
        X_train, Y_train = X.iloc[train_index,:], Y.iloc[train_index]
        X_test, Y_test = X.iloc[test_index,:], Y.iloc[test_index]
        
        # Using all data, ignore CV
        cent, var = baseline.fit(Y)
       
        row = [ds.name, iteration, cent, var]

        
        output_df = output_df.append( pd.DataFrame( [row], columns = out_cols ) )


# Save results as file
# Each iteration just in case
output_df.to_csv(output_file, index=False)
