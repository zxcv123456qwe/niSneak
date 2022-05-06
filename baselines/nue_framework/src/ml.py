## Imports
# Libraries
from cgi import test
import numpy as np
import pandas as pd
from sklearn import preprocessing
import argparse

# Databases of techniques
from map import dataset_db as ds_db
from map import transformation_db as dt_db
from map import selection_db as as_db
from map import learning_db as mlt_db
from map import optimization_db as pt_db
from map import baseline_db as bl_db
from map import validation_db as cv_db
from map import metric_db as em_db
from map import preprocessing_db as pp_db

from sklearn.metrics import make_scorer
from reading import Loader
from validation import LoaderCV
from evaluation import Evaluation
from transformation import Preprocessing
from joblib import Memory
from shutil import rmtree
from tempfile import mkdtemp
from selection import NumericalSelector
from transformation import FillImputer, SimplerImputer, KNNImputerDF
from transformation import OneHotEncoding
from evaluation import get_pareto_front, get_pareto_front_zitler, evaluate, \
    get_metrics_dataset, get_all_scorers, get_metrics_by_name
from utils import get_problem_type

from sklearn.pipeline import Pipeline
from pipeline import FeatureJoin
import datetime
import time

# Ingnore convergence warnings
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
# simplefilter("ignore", category=ConvergenceWarning)

## Argument parsing
parser = argparse.ArgumentParser(description='Machine learning and optimization framework')
parser.add_argument('config_dir', type=str,
                    help='Directory inside config/ with settings to run')
args = parser.parse_args()


## Prelude: Setup of the framework

# Load configuration
config_dir = f"config/{args.config_dir}/"
FW, DS, PP, DT, AS, PT, LA, EM = Loader(config_dir).load_config()
datasets = ds_db.get(DS)
preprocessing = pp_db.get(PP)
data_transformation = dt_db.get(DT)
as_techniques = as_db.get(AS)
pt_techniques = pt_db.get(PT)
ml_techniques = mlt_db.get(LA)
metrics = em_db.get(EM)

# Mode
# Type of problem to be solved
mode = get_problem_type(FW["problem"][0] if "problem" in FW.keys() else "classification")

# Pre processing
prep = Preprocessing( **(FW["preprocessing"] if "preprocessing" in FW.keys() else {}) )

# Cross Validation
# Defaults to 80:20 train test split
cv = cv_db[ FW["cv"][0] if "cv" in FW.keys() else "traintestsplit" ]( **(FW["cv"][1] if "cv" in FW.keys() else {"n_splits":1, "test_size":0.2}) )

# Pareto front
# Defaults to calculation by binary domination
pf = FW["pareto"][0] if "pareto" in FW.keys() else "binary"

# Output dataframe
right_now = datetime.datetime.now()

# out_cols = np.append(np.append(["DS", "Iteration", "DP", "AS", "PT", "LA", "PT metric", "Duration (s)"], metric_names), ["Models built", "Parameters"])
# output_df = pd.DataFrame(columns=out_cols)
output_df = None
output_file = "best-" + right_now.strftime('%Y-%m-%d_%H-%M-%S') + ".csv"

# Hyper-parameter output df
output_tuning_df = None
output_tuning_file = "hpt-" + right_now.strftime('%Y-%m-%d_%H-%M-%S') + ".csv"

# Pareto-frint output df
output_pareto_train_df = None
output_pareto_train_file = "train-" + right_now.strftime('%Y-%m-%d_%H-%M-%S') + ".csv"

# Pareto-frint output df
output_pareto_test_df = None
output_pareto_test_file = "test-" + right_now.strftime('%Y-%m-%d_%H-%M-%S') + ".csv"

# Dataset Loop
for n_ds, ds in enumerate(datasets):
    ds.set_datapath("data/")
    dataframe = ds.get_dataframe()

    # Data pre-processing
    # dataframe = prep.process(dataframe)
    X = dataframe[ dataframe.columns.difference([ds.predict]) ]
    Y = dataframe[ ds.predict ]
    # Type of problem: classification or regression
    problem = mode if ds.problem == "None" else ds.problem
    
    if type(cv) == LoaderCV:
        cv.set_data( ds.id )
        cv.set_config( config_dir )
    
    # Cross validation
    for iteration, (train_index, test_index) in enumerate(cv.split(dataframe)):        
        
        # Pre-processing
        for ppt in preprocessing:
        
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
                                
                                pst_scoring = f"({  pst.parameters['scoring'] if type(pst.parameters['scoring']) == str else ', '.join(pst.parameters['scoring'])   })" if 'scoring' in pst.parameters else ''
                                
                                print("-"*30)
                                print( current_time.strftime("%d/%m/%Y %H:%M:%S") )
                                print(f"Dataset: {ds.name} ({problem}) [{n_ds+1}/{len(datasets)}]")
                                print(f"Current iteration: {iteration+1}/{cv.get_n_splits(dataframe)}")
                                print(f"PP: {ppt.name}")
                                print(f"DT: {dtt.name}")
                                print(f"AS: {ast.name}")
                                print(f"LA: {mlt.name}")
                                print(f"PT: {pst.name} {pst_scoring}")
                                print("-"*30)
                                
                                # Get metrics that are applicable only to this dataset
                                ds_metrics = get_metrics_dataset(ds, metrics, problem)
                                metric_names = [ m.name for m in metrics ]
                                
                                # Get dataset partitions
                                # Re-generate if they get overwritten, unlikely
                                X_train, Y_train = X.iloc[train_index,:], Y.iloc[train_index]
                                X_test, Y_test = X.iloc[test_index,:], Y.iloc[test_index]
                                #print(train_index, test_index)
                                
                                # Apply pre-processing
                                # Just on training
                                pp_param = ppt.parameters.copy()
                                if pp_param.get("secondary") is True:
                                    pp_param.pop("secondary")
                                    pp_param["features"] = ds.secondary
                                pptt = ppt.pp_class( **pp_param )
                                X_train, Y_train = pptt.fit_transmute(X_train, Y_train)
                                
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
                                
                                model = mlt.get_class(problem)(**ml_param)
                                
                                # if AS is wrapper, use model as estimator
                                if "wrapper" in as_param.keys():
                                    if as_param["wrapper"] == True:
                                        as_param["estimator"] = model
                                    del as_param["wrapper"]
                                    
                                
                                astt = ast.as_class(**as_param)
                                
                                # Create pipeline to select attributes + fit model
                                location = mkdtemp()
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
                                # We also support multi-objective optimization
                                
                                multiobj_optim = False
                                pt_parameters = pst.parameters.copy()
                                if "scoring" in pt_parameters.keys():
                                    met_name = pt_parameters["scoring"]
                                    
                                    # We now convert the scoring from name to a function
                                    # Or to a dict of functions in the case of multi-objective
                                    opt_metrics = get_metrics_dataset(ds, metrics, problem, names = met_name)
                                    new_scoring = get_all_scorers( opt_metrics )
                                    multiobj_optim = len(new_scoring) > 1
                                    
                                    # If only one metric, we dont need dict
                                    if not(multiobj_optim):
                                        new_scoring = list(new_scoring.values())[0]
                                    
                                    pt_parameters["scoring"] = new_scoring
                                
                                # If the scorer is multiobjective
                                # We must define the "main" metric
                                # These will be priorized by tuners
                                if multiobj_optim:
                                    # If it is not manually defined
                                    # Use the first metric
                                    if "refit" not in pt_parameters:
                                        pt_parameters["refit"] = list(pt_parameters["scoring"].keys())[0]
                                    else:
                                        # If refit is not on metrics, we add it
                                        refit_name = pt_parameters["refit"]
                                        if refit_name not in pt_parameters["scoring"].keys():
                                            pt_parameters["scoring"][refit_name] = get_metrics_by_name(metrics, [refit_name])[0].make_scorer()
                                
                                search = pst.pt_class( pipe, search_space, **pt_parameters )
                                
                                start_time = time.time()
                                
                                search.fit(X_train, Y_train)
                                models_built = len( search.cv_results_["params"] ) * search.n_splits_
                                
                                # If we are using multi-objective, get pareto front
                                # Otherwise, just use the best parameters
                                best_params = []
                                if multiobj_optim:
                                    if pf == "binary":
                                        pareto_front = get_pareto_front( search.cv_results_, list(pt_parameters["scoring"].values()) )
                                    elif pf == "zitler":
                                        pareto_front = get_pareto_front_zitler( search.cv_results_, list(pt_parameters["scoring"].values()) )
                                    best_params = [ search.cv_results_["params"][i] for i in pareto_front ]
                                    multiobj_res_df = None # Separate dataset for pareto front
                                    duration_pareto = 0 # Duration counter
                                    models_built_pareto = 0 # Models built counter
                                    
                                    # Get pareto front in training set
                                    for idx, current_params in zip(pareto_front, best_params):
                                        results = dict( [ (met.name, search.cv_results_["mean_test_" + met.name][idx])  for met in pt_parameters["scoring"].values() ] )
                                        
                                        scoring = pst.parameters["scoring"] if "scoring" in pst.parameters.keys() else "None"
                                        row = {"DS": ds.name,
                                            "Iteration": iteration,
                                            "PP" : ppt.name,
                                            "DT" : dtt.name,
                                            "AS" : ast.name,
                                            "LA" : mlt.name,
                                            "PT" : pst.name,
                                            "PT scoring" : scoring,
                                            "Duration" : search.cv_results_["mean_fit_time"][idx] * search.cv,
                                            "Params" : str(search.best_params_),
                                            **results
                                        }
                                        row = dict([ (k, [v]) for k, v in row.items() ])
                                        result_frame = pd.DataFrame.from_dict(row)
                                        if output_pareto_train_df is None:
                                            output_pareto_train_df = result_frame
                                        else:
                                            output_pareto_train_df = pd.concat( [output_pareto_train_df, result_frame] )
                                        output_pareto_train_df.to_csv(output_pareto_train_file, index=False)
                                else:
                                    best_params = [ search.best_params_ ]
                                
                                for current_params in best_params:
                                    # Re-fit using best parameters
                                    pipe.set_params(**current_params)
                                    pipe.fit( X_train, Y_train )
                                    
                                    # Test the model
                                    prediction = pipe.predict(X_test)
                                    
                                    end_time = time.time()
                                    duration = end_time - start_time
                                    
                                    results = evaluate(Y_test, prediction, X_test, pipe, ds_metrics)
                                    
                                    # print(Y_test, prediction)
                                    
                                    # Output results
                                    # Add results to output file
                                    # np.append(np.append(["DS", "Iteration", "DP", "AS", "PT", "LA", "PT metric", "Duration (s)"], metric_names), ["Models built", "Parameters"])
                                    scoring = pst.parameters["scoring"] if "scoring" in pst.parameters.keys() else "None"
                                    row = {"DS": ds.name,
                                        "Iteration": iteration,
                                        "PP" : ppt.name,
                                        "DT" : dtt.name,
                                        "AS" : ast.name,
                                        "LA" : mlt.name,
                                        "PT" : pst.name,
                                        "PT scoring" : scoring,
                                        "Duration" : duration,
                                        "Params" : str(search.best_params_),
                                        **results
                                    }
                                    row = dict([ (k, [v]) for k, v in row.items() ])
                                    result_frame = pd.DataFrame.from_dict(row)
                                    
                                    # Save results as file
                                    # Each iteration just in case
                                    if multiobj_optim:
                                        if output_pareto_test_df is None:
                                            output_pareto_test_df = result_frame
                                        else:
                                            output_pareto_test_df = pd.concat( [output_pareto_test_df, result_frame] )
                                        output_pareto_test_df.to_csv(output_pareto_test_file, index=False)
                                        duration_pareto += duration
                                        models_built_pareto += models_built
                                        
                                        # Store metrics in another frame to calculate aggregation for front
                                        results1 = dict([ (k, [v]) for k, v in results.items() ])
                                        front_row = pd.DataFrame.from_dict(results1)
                                        if multiobj_res_df is None:
                                            multiobj_res_df = front_row
                                        else:
                                            multiobj_res_df = pd.concat( [multiobj_res_df, front_row] )
                                        
                                    else:
                                        if output_df is None:
                                            output_df = result_frame
                                        else:
                                            output_df = pd.concat( [output_df, result_frame] )
                                        
                                        output_df.to_csv(output_file, index=False)
                                    
                                # Hyper-parameter dataframe
                                hyper_dict = search.cv_results_
                                df_size = len( hyper_dict["params"] )
                                hyper_dict["DS"] = np.repeat(ds.name, df_size)
                                hyper_dict["Iteration"] = np.repeat(iteration, df_size)
                                hyper_dict["PP"] = np.repeat(ppt.name, df_size)
                                hyper_dict["DP"] = np.repeat(dtt.name, df_size)
                                hyper_dict["AS"] = np.repeat(ast.name, df_size)
                                hyper_dict["PT"] = np.repeat(pst.name, df_size)
                                hyper_dict["LA"] = np.repeat(mlt.name, df_size)
                                
                                hyper_frame = pd.DataFrame.from_dict(hyper_dict)
                                
                                if output_tuning_df is None:
                                    output_tuning_df = hyper_frame
                                else:
                                    output_tuning_df = pd.concat( [output_tuning_df, hyper_frame] )
                                
                                # Uncomment if you need it
                                # Files grow too large
                                output_tuning_df.to_csv(output_tuning_file, index=False)
                                
                                # If we are on a pareto front, save average results
                                if multiobj_optim:
                                    # Average metrics of the front
                                    metrics_front = multiobj_res_df.mean(axis = 0)
                                    
                                    # Construct row in a format similar to normal frame
                                    row = {"DS": ds.name,
                                        "Iteration": iteration,
                                        "PP" : ppt.name,
                                        "DT" : dtt.name,
                                        "AS" : ast.name,
                                        "LA" : mlt.name,
                                        "PT" : pst.name,
                                        "PT scoring" : scoring,
                                        "Duration" : duration_pareto,
                                        "Models built" : models_built,
                                        "Best params" : "pareto front",
                                        **metrics_front
                                    }
                                    row = dict([ (k, [v]) for k, v in row.items() ])
                                    result_frame = pd.DataFrame.from_dict(row)
                                    
                                    # Save to dataframe
                                    if output_df is None:
                                        output_df = result_frame
                                    else:
                                        output_df = pd.concat( [output_df, result_frame] )
                                        
                                        output_df.to_csv(output_file, index=False)
                            
                                # Clear cache
                                rmtree(location)

