from sklearn.model_selection._search import BaseSearchCV
from hyperopt import fmin, tpe, hp
import numpy as np
from optimization import BaseOptimizer

# TPE algorithm
# Using the implementation of the hyperopt repository
# http://hyperopt.github.io/hyperopt/
# Adapted to the scikit learn BaseSearchCV class
class TPECV(BaseOptimizer):
    
    def __init__(self, estimator, search_space, budget, *, scoring=None, n_jobs=None,
                 refit=True, cv=None, verbose=0,
                 pre_dispatch='2*n_jobs', error_score=np.nan, return_train_score=True, **kwargs):
        
        super().__init__(
            estimator=estimator, scoring=scoring,
            n_jobs=n_jobs, refit=refit, cv=cv, verbose=verbose,
            pre_dispatch=pre_dispatch, error_score=error_score,
            return_train_score=return_train_score)
        
        self.search_space = search_space
        self.budget = budget
    
    def _run_search(self, evaluate_candidates):
        self.multimetric_ = isinstance(self.scoring, dict)
        self.rank_test_name_ = "rank_test_" + self.refit if self.multimetric_ else "rank_test_score"
        self.mean_test_name_ = "mean_test_" + self.refit if self.multimetric_ else "mean_test_score"
        self.scoring_sign_ = self.scoring[self.refit]._sign if self.multimetric_ else self.scoring._sign
        
        # Search space to hyperopt format
        space = dict([ (k, hp.choice( k, v ) ) for k, v in self.search_space.items() ])
        
        # Objective function for hyperopt format
        def eval_function( params ):
            result = evaluate_candidates( [params] )
            loss = result[self.mean_test_name_][-1] * self.scoring_sign_ * -1
            return loss
        
        fmin( fn = eval_function,
             space = space,
             algo = tpe.suggest,
             max_evals = self.budget,
             verbose = self.verbose != 0)
        
        