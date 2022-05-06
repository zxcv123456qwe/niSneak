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
# cv = cv_db[ FW["cv"][0] if "cv" in FW.keys() else "traintestsplit" ]( **(FW["cv"][1] if "cv" in FW.keys() else {"n_splits":1, "test_size":0.2}) )


# Output dataframe
out_cols = np.append(np.append(["DS", "Iteration", "DP", "AS", "PT", "LA", "Duration (s)"], EM), ["Parameters"])
output_df = pd.DataFrame(columns=out_cols)
output_file = "result-" + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + ".csv"

# Dataset Loop
for ds in datasets:
    ds.set_datapath("data/")
    
    df_train = ds.get_train().get_dataframe()
    df_test = ds.get_test().get_dataframe()
    
    # Data pre-processing
    # dataframe = prep.process(dataframe)
    X_train = df_train[ df_train.columns.difference([ds.predict]) ]
    Y_train = df_train[ ds.predict ]
    X_test = df_test[ df_test.columns.difference([ds.predict]) ]
    Y_test = df_test[ ds.predict ]
    
    
    # Learning scheme
    # Data Transformation
    for dtt in data_transformation:
        
        # Attribute selection
        for ast in as_techniques:
            astp = ast.parameters
            
            # Learning algorithm
            for mlt in ml_techniques:
                
                # Parameter tuning
                for pst in pt_techniques:
                        current_time = datetime.datetime.now()    
                    
                        print("-"*30)
                        print( current_time.strftime("%d/%m/%Y %H:%M:%S") )
                        print("Dataset: %s" % ds.name)
                        print("Current iteration: TEST")
                        print("DT: %s" % dtt.name)
                        print("AS: %s" % ast.name)
                        print("LA: %s" % mlt.name)
                        print("PT: %s" % pst.name)
                        print("-"*30)
                        
                        # Depending on algorithm
                        # Select set of parameters to evaluate
                        # Tuner acts on the pipeline
                        
                        # Because of pipelines
                        # A new dictionary must be created
                        # Also, separate non-search parameters
                        search_space = {}
                        dt_param = {}
                        as_param = {}
                        ml_param = {}
                        for key, val in dtt.parameters.items():
                            if type(val) == list:
                                search_space["dt__numerical__transform__" + key] = val
                            else:
                                dt_param[key] = val
                        for key, val in ast.parameters.items():
                            if type(val) == list:
                                search_space["as__" + key] = val
                            else:
                                as_param[key] = val
                        for key, val in mlt.parameters.items():
                            if type(val) == list:
                                search_space["la__" + key] = val
                            else:
                                ml_param[key] = val
                        
                        # Finally, train the scheme
                        dttt = dtt.dt_class(**dt_param)
                        
                        model = mlt.ml_class(**ml_param)
                        
                        # if AS is wrapper, use model as estimator
                        if "wrapper" in as_param.keys():
                            if as_param["wrapper"] == True:
                                as_param["estimator"] = model
                            del as_param["wrapper"]
                            
                        
                        astt = ast.as_class(**as_param)
                        
                        # Create pipeline to select attributes + fit model
                        location = 'cachedir'
                        memory = Memory(location=location, verbose=0)
                        pipe = Pipeline([
                                ('dt', FeatureJoin(transformer_list=[
                                    ('numerical', Pipeline([
                                            ('select', NumericalSelector(True)),
                                            ('imputation', KNNImputerDF(missing_values = np.nan, n_neighbors=1)),
                                            ('transform', dttt)
                                        ])),
                                    ('categorical', Pipeline([
                                            ('select', NumericalSelector(False)),
                                            ('imputation', FillImputer()),
                                            ('transform', OneHotEncoding(sparse=False, handle_unknown='ignore'))
                                        ]))
                                    ])),
                                ('as', astt),
                                ('la', model)
                                 ],memory=memory)
                        
                        # Use requested scoring
                        pt_parameters = pst.parameters.copy()
                        if "scoring" in pt_parameters.keys():
                            met_name = pt_parameters["scoring"]
                            pt_parameters["scoring"] = make_scorer( eva.get_function(met_name), greater_is_better = eva.get_greater_is_better(met_name) )
                        
                        search = pst.pt_class( pipe, search_space, **pt_parameters )
                        
                        start_time = time.time()
                        
                        search.fit(X_train, Y_train)
                        
                        # Re-fit using best parameters
                        pipe.set_params(**search.best_params_)
                        pipe.fit( X_train, Y_train )
                        
                        # Test the model
                        prediction = pipe.predict(X_test)
                        
                        end_time = time.time()
                        duration = end_time - start_time
                        
                        # Clear cache
                        memory.clear(warn=False)
                        rmtree(location)
                        
                        metrics = eva.evaluate(Y_test, prediction)
                        
                        # Output results
                        # Add results to output file
                        row = [ds.name, "TEST", dtt.name, ast.name, pst.name, mlt.name]
                        row.append( "%.4f" % duration )
                        
                        for metric in metrics.keys():
                            row.append( "%.4f" % metrics[metric] )
                        
                        row.append( search.best_params_ )
                        
                        output_df = output_df.append( pd.DataFrame( [row], columns = out_cols ) )
            
            
                        # Save results as file
                        # Each iteration just in case
                        output_df.to_csv(output_file, index=False)
