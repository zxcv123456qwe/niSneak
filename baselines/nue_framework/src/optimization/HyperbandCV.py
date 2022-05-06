from sklearn.model_selection._search import BaseSearchCV
import numpy as np
from abc import ABCMeta
from optimization import grid_to_bounds, grid_types, cast_parameters, aggregate_dict, random_population
import pandas as pd
from sklearn.model_selection import train_test_split
from optimization import Hyperband
from optimization import BaseOptimizer

# Hyperband algorithm
# Using the implementation of the Hyperband repository
# https://github.com/zygmuntz/hyperband
# Adapted to the scikit learn BaseSearchCV class

class HyperbandCV(BaseOptimizer):
    
    def __init__(self, estimator, search_space, budget, eta, *, scoring=None, n_jobs=None,
                 refit=True, cv=None, verbose=0,
                 pre_dispatch='2*n_jobs', error_score=np.nan, return_train_score=True, **kwargs):
        
        super().__init__(
            estimator=estimator, scoring=scoring,
            n_jobs=n_jobs, refit=refit, cv=cv, verbose=verbose,
            pre_dispatch=pre_dispatch, error_score=error_score,
            return_train_score=return_train_score)
        
        self.search_space = search_space
        self.budget = budget
        self.eta = eta
    
    def _run_search(self, evaluate_candidates):
        self.multimetric_ = isinstance(self.scoring, dict)
        self.rank_test_name_ = "rank_test_" + self.refit if self.multimetric_ else "rank_test_score"
        self.mean_test_name_ = "mean_test_" + self.refit if self.multimetric_ else "mean_test_score"
        self.scoring_sign_ = self.scoring[self.refit]._sign if self.multimetric_ else self.scoring._sign
        
        # Hyperband needs an evaluation function
        # As well as a random parameter generator
        
        # Determine seach bounds
        bounds = grid_to_bounds(self.search_space)
        categories = dict( [ (k,v) for k,v in self.search_space.items() if k not in bounds.keys() ] )
        types = grid_types(self.search_space)
        # Random individual
        random_individual = lambda : random_population(bounds, list(types.values()), categories, 1)[0]
        
        # Evaluation function returns loss
        # We invert the value
        def eval_function( n_iter, params ):
            result = evaluate_candidates( [params] )
            loss = result[self.mean_test_name_][-1] * self.scoring_sign_ * -1
            return { "loss" : loss, "log_loss" : loss, "auc" : 0 }
        
        # Instantiate searcher object
        self.hb_ = Hyperband( random_individual, eval_function, max_iter = self.budget, eta = self.eta )
        
        # Run Hyper Bandit
        self.hb_.run( verbose = False )
        
