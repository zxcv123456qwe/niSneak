# Libraries
import numpy as np
import pandas as pd

# TODO fix imports
from database.dataset_db import dataset_db
from database.dt_db import dt_db
from database.bl_db import bl_db
from sklearn.metrics import make_scorer
from comp.Loader import Loader
from comp.Evaluation import Evaluation
from comp.Preprocessing import Preprocessing
from joblib import Memory
from shutil import rmtree
from comp.Selector import NumericalSelector
from comp.Imputer import FillImputer, SimplerImputer, KNNImputerDF
from comp.DataTransformation import OneHotEncoding
from dim.dim import Solver


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

# Pre processing
# prep = Preprocessing(remove_missing_columns = 0.25, missing_value_handling  = Missing_Value_Handling.MEAN)

prep = Preprocessing( **(FW["preprocessing"] if "preprocessing" in FW.keys() else {}) )

# Baseline
baseline = bl_db[ FW["baseline"][0] if "baseline" in FW.keys() else "none" ]

# Evaluation metrics
eva = Evaluation(EM, baseline)

# Cross Validation
# Defaults to 80:20 train test split
# cv = cv_db[ FW["cv"][0] if "cv" in FW.keys() else "traintestsplit" ]( **(FW["cv"][1] if "cv" in FW.keys() else {"n_splits":1, "test_size":0.2}) )


# Output dataframe
out_cols = ["DS", "instances", "features", "features (prep)", "features (num)", "features (non-num)", "L1", "L2"]
output_df = pd.DataFrame(columns=out_cols)
output_file = "dimensionality-" + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + ".csv"

# Dataset Loop
for ds in datasets:
    ds.set_datapath("data/")
    
    dataframe = ds.get_dataframe()
    
    # Data pre-processing
    # dataframe = prep.process(dataframe)
    X = dataframe[ dataframe.columns.difference([ds.predict]) ]
    Y = dataframe[ ds.predict ]
    
    # Convert everything to numerical
    # Inpute missing values
    pipe = Pipeline([
        ('dt', FeatureJoin(transformer_list=[
            ('numerical', Pipeline([
                    ('select', NumericalSelector(True)),
                    ('imputation', KNNImputerDF(missing_values = np.nan, n_neighbors=1))
                ])),
            ('categorical', Pipeline([
                    ('select', NumericalSelector(False)),
                    ('imputation', FillImputer()),
                    ('transform', OneHotEncoding(sparse=False, handle_unknown='ignore'))
                ]))
            ]))
         ])
    
    df_orig = pd.concat([X , Y], axis=1)
    X2 = pipe.fit_transform( X, Y )
    
    df = pd.concat([X2 , Y], axis=1)
    
    num_features = NumericalSelector(True).fit_transform(df_orig).shape[1]
    non_num_features = NumericalSelector(False).fit_transform(df_orig).shape[1]
    
    # Calculate dimensionalities
    solver = Solver(df)
    l1 = solver.show_curve("-10:10:100", version = 1)
    l2 = solver.show_curve("-10:10:100", version = 2)
    
    # Output results
    # Add results to output file
    row = [ds.name, df_orig.shape[0], df_orig.shape[1], df.shape[1], num_features, non_num_features, l1, l2]
    
    output_df = output_df.append( pd.DataFrame( [row], columns = out_cols ) )


    # Save results as file
    # Each iteration just in case
    output_df.to_csv(output_file, index=False)
