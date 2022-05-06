## Imports
# Libraries
import numpy as np
import pandas as pd

# Databases of techniques
from map import model_db as mo_db
from map import optimization_db as pt_db

from sklearn.metrics import make_scorer
from reading import Loader
from evaluation import get_pareto_front, evaluate, \
    get_metrics_dataset, get_all_scorers, get_metrics_by_name
from utils import get_problem_type

import datetime
import time

# Ingnore convergence warnings
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

## Prelude: Setup of the framework

# Load configuration
config = "splmodel"
FW, MO, PT = Loader("config/" + config + "/").load_config_model()
models = mo_db.get(MO)
pt_techniques = pt_db.get(PT)

# Iterations to execute
iterations = int(FW["iterations"][0]) if "iterations" in FW.keys() else 1

# Output dataframe
right_now = datetime.datetime.now()

# Hyper-parameter output df
output_tuning_df = None
output_tuning_file = "result-" + config + "-model-hpt-" + right_now.strftime('%Y-%m-%d_%H-%M-%S') + ".csv"

# Pareto-frint output df
output_pareto_df = None
output_pareto_file = "result-" + config + "-model-pareto-" + right_now.strftime('%Y-%m-%d_%H-%M-%S') + ".csv"

# Model Loop
for n_model, moo in enumerate(models):
    
    
    for iteration in range(iterations):
    
        # Parameter tuning
        for pst in pt_techniques:
                current_time = datetime.datetime.now()
                
                print("-"*30)
                print( current_time.strftime("%d/%m/%Y %H:%M:%S") )
                print(f"Model: {moo.name} [{n_model+1}/{len(models)}]")
                print(f"Current iteration: {iteration+1}/{iterations}")
                print(f"PT: {pst.name}")
                print("-"*30)
                
                # Search space
                mo_param = {}
                search_space = {}
                for key, val in moo.parameters.items():
                    if type(val) == list:
                        search_space[key] = val
                    else:
                        mo_param[key] = val
                
                # Model
                model = moo.mo_class( **mo_param )
                
                # Get metrics that are applicable only to this dataset
                metrics = model.get_scorings()
                metric_names = list( metrics.keys() )
                
                # Use requested scoring
                # We also support multi-objective optimization
                
                multiobj_optim = False
                pt_parameters = pst.parameters.copy()
                multiobj_optim = len(metrics) > 1
                pt_parameters["scoring"] = metrics.copy()
                
                search = pst.pt_class( model, search_space, **pt_parameters )
                
                start_time = time.time()
                
                search.optimize_model()
                models_built = len( search.cv_results_["params"] )
                # print( search.cv_results_ )
                # print( pt_parameters["scoring"] )
                
                # If we are using multi-objective, get pareto front
                # Otherwise, just use the best parameters
                best_params = []
                pareto_front = get_pareto_front( search.cv_results_, pt_parameters["scoring"].values() )
                best_params = [ search.cv_results_["params"][i] for i in pareto_front ]
                multiobj_res_df = None # Separate dataset for pareto front
                duration_pareto = 0 # Duration counter
                models_built_pareto = 0 # Models built counter
                
                # Get pareto front in training set
                for idx, current_params in zip(pareto_front, best_params):
                    results = dict( [ (met.name, search.cv_results_["mean_test_" + met.name][idx])  for met in pt_parameters["scoring"].values() ] )
                    
                    scoring = pst.parameters["scoring"] if "scoring" in pst.parameters.keys() else "None"
                    row = {"MO": moo.name,
                        "Iteration": iteration,
                        "PT" : pst.name,
                        "Duration" : search.cv_results_["mean_fit_time"][idx],
                        **results
                    }
                    row = dict([ (k, [v]) for k, v in row.items() ])
                    result_frame = pd.DataFrame.from_dict(row)
                    if output_pareto_df is None:
                        output_pareto_df = result_frame
                    else:
                        output_pareto_df = pd.concat( [output_pareto_df, result_frame] )
                    output_pareto_df.to_csv(output_pareto_file, index=False)
                
                
                # Hyper-parameter dataframe
                hyper_dict = search.cv_results_
                df_size = len( hyper_dict["params"] )
                hyper_dict["MOO"] = np.repeat(moo.name, df_size)
                hyper_dict["Iteration"] = np.repeat(iteration, df_size)
                hyper_dict["PT"] = np.repeat(pst.name, df_size)
                
                hyper_frame = pd.DataFrame.from_dict(hyper_dict)
                
                if output_tuning_df is None:
                    output_tuning_df = hyper_frame
                else:
                    output_tuning_df = pd.concat( [output_tuning_df, hyper_frame] )
                
                # Uncomment if you need it
                # Files grow too large
                output_tuning_df.to_csv(output_tuning_file, index=False)
                