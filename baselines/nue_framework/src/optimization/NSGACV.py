from sklearn.model_selection._search import BaseSearchCV
import numpy as np
from optimization import grid_to_bounds, grid_types, cast_parameters, zip_one
from deap import base, creator, tools
import random
import warnings
from optimization import BaseOptimizer

# NSGA-II and NSGA-III algorithm
# DEAP genetic algorithms adapted to optimization
# https://github.com/DEAP/deap
# Adapted to the scikit learn BaseSearchCV class
class NSGACV(BaseOptimizer):
    
    def __init__(self, estimator, search_space, version,  mutation_rate, crossover_rate,
                 population_size, iterations, *, p = 4, scale = 1.0,  scoring=None, n_jobs=None,
                 refit=True, cv=None, verbose=0,
                 pre_dispatch='2*n_jobs', error_score=np.nan, return_train_score=True, **kwargs):
        
        super().__init__(
            estimator=estimator, scoring=scoring,
            n_jobs=n_jobs, refit=refit, cv=cv, verbose=verbose,
            pre_dispatch=pre_dispatch, error_score=error_score,
            return_train_score=return_train_score)
        
        self.search_space = search_space
        self.version = version
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population_size = population_size
        self.iterations = iterations
        
         # Only used in NSGA-III
        self.p = p
        self.scale = scale
    
    def _run_search(self, evaluate_candidates):
        self.multimetric_ = isinstance(self.scoring, dict)
        self.rank_test_name_ = "rank_test_" + self.refit if self.multimetric_ else "rank_test_score"
        self.mean_test_name_ = "mean_test_" + self.refit if self.multimetric_ else "mean_test_score"
        self.scoring_sign_ = self.scoring[self.refit]._sign if self.multimetric_ else self.scoring._sign
        
        # Force our scorer to become a dict
        scoring = self.scoring if isinstance(self.scoring, dict) else { self.scoring.name : self.scoring }
        
        bounds = grid_to_bounds(self.search_space)
        types = grid_types(self.search_space)
        types = dict( [ (key, val) for key, val in types.items() if key in bounds.keys() ] ) # Only for bounded types
        dimensions = len(bounds)
        
        # Objectives
        weights = [ float(m._sign) for m in scoring.values() ]
        with warnings.catch_warnings(): # Ignore the creation warnings
            warnings.simplefilter("ignore")
            creator.create("FitnessMulti", base.Fitness, weights = tuple(weights) )
            creator.create("Individual", list, fitness=creator.FitnessMulti)
        
        toolbox = base.Toolbox()
        
        # Attributes for individual
        for k in self.search_space.keys():
            if k not in types.keys():
                # str
                toolbox.register(k, random.choice, self.search_space[k])
            else:
                fun = random.randint if types[k] in (int, np.int32) else random.uniform
                toolbox.register(k, fun, bounds[k][0], bounds[k][1])
        
        # Individual, very funky
        toolbox.register("individual", tools.initCycle, creator.Individual,
                         [toolbox.__getattribute__(k) for k in self.search_space.keys()] )
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        # Define evaluation function
        def eval_one(individual):
            individual = zip_one(self.search_space.keys(), individual)
            cache = evaluate_candidates([individual])
            idx = cache["params"].index( individual )
            if not self.multimetric_:
                return cache[ self.mean_test_name_ ][idx],
            else:
                return tuple([ cache[ f'mean_test_{m}' ][idx] for m in scoring.keys() ])
        
        toolbox.register("evaluate", eval_one)
        
        bounds_sim = [ bounds[k] if k in bounds.keys() else self.search_space[k] for k in self.search_space.keys() ]
        types_sim = [ types[k] if k in types.keys() else str for k in self.search_space.keys() ]
        
        # Define our methods
        toolbox.register("mate", cxSimulatedBinaryBounded, bounds=bounds_sim, types=types_sim, eta=20.0)
        toolbox.register("mutate", mutPolynomialBounded, bounds=bounds_sim, types=types_sim,
                         eta=20.0, indpb=1.0/dimensions)
        if self.version == 2:
            toolbox.register("select", tools.selNSGA2)
        if self.version == 3:
            ref = tools.uniform_reference_points(len( scoring ), self.p, self.scale)
            toolbox.register("select", tools.selNSGA3, ref_points = ref )
        
        # And now, we start the actual algorithm
        # Based off https://github.com/DEAP/deap/blob/master/examples/ga/nsga2.py
        # NSGA-III is similar, slightly different mutation/crossover setup
        pop = toolbox.population(n=self.population_size)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        # This is just to assign the crowding distance to the individuals
        # no actual selection is done
        pop = toolbox.select(pop, len(pop))
        offspring = pop
        
        # Begin the generational process
        # -1 because of initial population
        for gen in range(1, self.iterations):
            # Vary the population
            if self.version == 2:
                offspring = tools.selTournamentDCD(pop, len(pop))
            offspring = [toolbox.clone(ind) for ind in offspring]
            
            for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
                if random.random() <= self.crossover_rate:
                    toolbox.mate(ind1, ind2)

                if random.random() <= self.mutation_rate:
                    toolbox.mutate(ind1)
                if random.random() <= self.mutation_rate:
                    toolbox.mutate(ind2)
                del ind1.fitness.values, ind2.fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Select the next generation population
            pop = toolbox.select(pop + offspring, self.population_size)
    
        # We are done
        

def cxSimulatedBinaryBounded(ind1, ind2, eta, bounds, types):
    """Executes a simulated binary crossover that modify in-place the input
    individuals. The simulated binary crossover expects :term:`sequence`
    individuals of floating point numbers.
    :param ind1: The first individual participating in the crossover.
    :param ind2: The second individual participating in the crossover.
    :param eta: Crowding degree of the crossover. A high eta will produce
                children resembling to their parents, while a small eta will
                produce solutions much more different.
    :param bounds: A :term:`python:sequence`, of which each element is
            another :term:`python:sequence`, being either the lower and upper
            bounds of the search space, or all possible values.
    :param types: A  :term:`python:sequence` of values that contains the data types
            of the attributes of each individual.
    :returns: A tuple of two individuals.
    This function uses the :func:`~random.random` function from the python base
    :mod:`random` module.
    .. note::
    This implementation is similar to the one implemented in the
    original NSGA-II C code presented by Deb.
    Moreover, modified from DEAP implementation to account for int and str
    """
    size = min(len(ind1), len(ind2))
    if len(bounds) < size:
        raise IndexError("bounds must be at least the size of the shorter individual: %d < %d" % (len(bounds), size))
    if len(types) < size:
        raise IndexError("up must be at least the size of the shorter individual: %d < %d" % (len(types), size))

    for i in range(size):
        if random.random() <= 0.5:
            if types[i] != str:
                xl = bounds[i][0]
                xu = bounds[i][1]
                # This epsilon should probably be changed for 0 since
                # floating point arithmetic in Python is safer
                if abs(ind1[i] - ind2[i]) > 1e-14:
                    x1 = min(ind1[i], ind2[i])
                    x2 = max(ind1[i], ind2[i])
                    rand = random.random()

                    beta = 1.0 + (2.0 * (x1 - xl) / (x2 - x1))
                    alpha = 2.0 - beta ** -(eta + 1)
                    if rand <= 1.0 / alpha:
                        beta_q = (rand * alpha) ** (1.0 / (eta + 1))
                    else:
                        beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))

                    c1 = 0.5 * (x1 + x2 - beta_q * (x2 - x1))

                    beta = 1.0 + (2.0 * (xu - x2) / (x2 - x1))
                    alpha = 2.0 - beta ** -(eta + 1)
                    if rand <= 1.0 / alpha:
                        beta_q = (rand * alpha) ** (1.0 / (eta + 1))
                    else:
                        beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))
                    c2 = 0.5 * (x1 + x2 + beta_q * (x2 - x1))

                    c1 = min(max(c1, xl), xu)
                    c2 = min(max(c2, xl), xu)

                    if random.random() <= 0.5:
                        ind1[i] = c2
                        ind2[i] = c1
                    else:
                        ind1[i] = c1
                        ind2[i] = c2
                    
                    # Cast to correct type
                    ind1[i] = types[i](ind1[i])
                    ind2[i] = types[i](ind2[i])
            else:
                # Strings
                # Just exchange them
                c2 = ind1[i]
                c1 = ind2[i]
                ind1[i] = c1
                ind2[i] = c2

    return ind1, ind2

def mutPolynomialBounded(individual, eta, bounds, types, indpb):
    """Polynomial mutation as implemented in original NSGA-II algorithm in
    C by Deb.
    :param individual: :term:`Sequence <sequence>` individual to be mutated.
    :param eta: Crowding degree of the mutation. A high eta will produce
                a mutant resembling its parent, while a small eta will
                produce a solution much more different.
    :param low: A value or a :term:`python:sequence` of values that
                is the lower bound of the search space.
    :param up: A value or a :term:`python:sequence` of values that
               is the upper bound of the search space.
    :returns: A tuple of one individual.
    Modified from DEAP implementation to account for int and str
    """
    size = len(individual)
    if len(bounds) < size:
        raise IndexError("bounds must be at least the size of the shorter individual: %d < %d" % (len(bounds), size))
    if len(types) < size:
        raise IndexError("up must be at least the size of the shorter individual: %d < %d" % (len(types), size))

    for i in range(size):
        if random.random() <= indpb:
            x = individual[i]
            if types[i] != str:
                xl = bounds[i][0]
                xu = bounds[i][1]
            
                delta_1 = (x - xl) / (xu - xl)
                delta_2 = (xu - x) / (xu - xl)
                rand = random.random()
                mut_pow = 1.0 / (eta + 1.)

                if rand < 0.5:
                    xy = 1.0 - delta_1
                    val = 2.0 * rand + (1.0 - 2.0 * rand) * xy ** (eta + 1)
                    delta_q = val ** mut_pow - 1.0
                else:
                    xy = 1.0 - delta_2
                    val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * xy ** (eta + 1)
                    delta_q = 1.0 - val ** mut_pow

                x = x + delta_q * (xu - xl)
                x = min(max(x, xl), xu)
                x = types[i](x)
            else:
                # Just pick something at random
                x = random.choice( bounds[i] )
            individual[i] = x
    return individual,