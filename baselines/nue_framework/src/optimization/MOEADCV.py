from sklearn.model_selection._search import BaseSearchCV
import numpy as np
from optimization import grid_to_bounds_str, grid_types, cast_parameters, zip_one
from pymoo.problems.functional import FunctionalProblem
from pymoo.factory import get_reference_directions
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.factory import get_crossover, get_mutation
from pymoo.optimize import minimize
from optimization import BaseOptimizer

# MOEA/D algorithm
# Using pymoo implementation
# https://pymoo.org/
# Adapted to the scikit learn BaseSearchCV class
class MOEADCV(BaseOptimizer):
    
    def __init__(self, estimator, search_space, mutation_rate, crossover_rate,
                 n_neighbors, iterations, prob_neighbor_mating,
                 *, n_partitions = 4, scoring=None, n_jobs=None,
                 refit=True, cv=None, verbose=0,
                 pre_dispatch='2*n_jobs', error_score=np.nan, return_train_score=True, **kwargs):
        
        super().__init__(
            estimator=estimator, scoring=scoring,
            n_jobs=n_jobs, refit=refit, cv=cv, verbose=verbose,
            pre_dispatch=pre_dispatch, error_score=error_score,
            return_train_score=return_train_score)
        
        self.search_space = search_space
        self.n_neighbors = n_neighbors
        self.prob_neighbor_mating = prob_neighbor_mating
        self.n_partitions = n_partitions
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.iterations = iterations
    
    def _run_search(self, evaluate_candidates):
        self.multimetric_ = isinstance(self.scoring, dict)
        self.rank_test_name_ = "rank_test_" + self.refit if self.multimetric_ else "rank_test_score"
        self.mean_test_name_ = "mean_test_" + self.refit if self.multimetric_ else "mean_test_score"
        self.scoring_sign_ = self.scoring[self.refit]._sign if self.multimetric_ else self.scoring._sign
        
        # Force our scorer to become a dict
        scoring = self.scoring if isinstance(self.scoring, dict) else { self.scoring.name : self.scoring }
        
        bounds = grid_to_bounds_str(self.search_space)
        types = grid_types(self.search_space)
        # types = dict( [ (key, val) for key, val in types.items() if key in bounds.keys() ] ) # Only for bounded types
        dimensions = len(bounds)
        
        # For now work with non-str parameters
        # Define evaluation function
        def eval_one(individual, metric):
            individual = list(individual)
            for i, (n, t) in enumerate(types.items()):
                if t in [str, np.character]:
                    individual[i] = self.search_space[n][ int(individual[i]) ]
            individual = cast_parameters(individual, list(types.values()))
            individual = zip_one(self.search_space.keys(), individual)
            cache = evaluate_candidates([individual])
            idx = cache["params"].index( individual )
            if not self.multimetric_:
                return cache[ self.mean_test_name_ ][idx] * self.scoring[metric]._sign * -1
            else:
                return cache[ f'mean_test_{metric}' ][idx]  * self.scoring[metric]._sign * -1
        
        # Create pymoo problem
        problem = FunctionalProblem(
            dimensions, # number of parameters to optimize
            list( map(lambda i: lambda x: eval_one( x, i ), self.scoring.keys()) ),
            xl = np.array([ x[0] for x in bounds.values() ]),
            xu = np.array([ x[1] for x in bounds.values() ]),
        )
        
        # Test the problem
        # pop = random_population( bounds, list(types.values()), {}, 5 )
        # pop = np.array([ list(x.values()) for x in pop ])
        # print(pop)
        # F = problem.evaluate(pop)
        # print(F)
        
        # Pymoo algorithm: MOEA/D
        ref_dirs = get_reference_directions("das-dennis", len(scoring), n_partitions=self.n_partitions)
        algo = MOEAD(
            ref_dirs,
            n_neighbors = self.n_neighbors,
            prob_neighbor_mating = self.prob_neighbor_mating,
            crossover = get_crossover("real_sbx", prob=self.crossover_rate, eta=20),
            mutation = get_mutation("real_pm", prob=self.mutation_rate, eta=20),
        )
        
        res = minimize(problem,
               algo,
               ('n_gen', self.iterations),
               verbose=False)
        
        