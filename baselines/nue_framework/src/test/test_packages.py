# From https://stackoverflow.com/questions/25827160/importing-correctly-with-pytest
import sys, os
sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/.."))
import pytest

# Uncomment when you have no idea what is failing
# Or if you need to test everything
# from preprocessing import *
# from baseline import *
# from dimensionality import *
# from evaluation import *
# from learning import *
# from pipeline import *
# from reading import *
# from selection import *
# from transformation import *
# from optimization import *
# from validation import *
# from utils import *


def test_imports():
    """
        Test Imports.
        Tests that every importable class works, and can be imported from the package.
        Should be updated as more classes are added.
    """

    # Util package
    from utils import ps
    from utils import get_problem_type
    
    # Preprocessing package
    from preprocessing import PreProcessor
    from preprocessing import FairSmoteSelector

    # Baseline package
    from baseline import Baseline
    from baseline import MARP0
    from baseline import MARP0LOO
    from baseline import MDARP0
    from baseline import Median

    # Dimensionality package
    from dimensionality import Solver

    # Evaluation/Metrics package
    from evaluation import Evaluation
    from evaluation import get_pareto_front
    from evaluation import Metric
    from evaluation import MetricX
    from evaluation import MetricFull
    from evaluation import get_metrics_problem
    from evaluation import evaluate
    from evaluation import get_all_scorers
    from evaluation import get_metrics_by_name
    from evaluation import get_metricx_list
    from evaluation import get_metrics_dataset
    from evaluation import AOD
    from evaluation import EOD
    from evaluation import SPD
    from evaluation import DI
    from evaluation import FR

    # Learning package
    from learning import Learner
    from learning import MLPReg

    # Pipeline package
    from pipeline import FeatureJoin

    # Reading/IO package
    from reading import Dataset
    from reading import Loader

    # Selection package
    from selection import FeatureSelection
    from selection import ColumnSelector
    from selection import NumericalSelector
    from selection import AttributeSelector
    from selection import VariantThreshold
    from selection import BorutaSelector
    from selection import DummySelector

    # Transformation package
    from transformation import DataTransformation
    from transformation import OneHotEncoding
    from transformation import StandardScaling
    from transformation import FillImputer
    from transformation import SimplerImputer
    from transformation import KNNImputerDF
    from transformation import Missing_Value_Handling
    from transformation import Preprocessing
    from transformation import MinMaxScaling

    # Optimization package
    from optimization import Optimizer
    from optimization import DefaultCV
    from optimization import BayesianOptimizationCV
    from optimization import DifferentialEvolutionCV
    from optimization import DodgeCV
    from optimization import GeneticAlgorithmCV
    from optimization import HarmonySearchCV
    from optimization import HyperbandCV
    from optimization import NeverGradCV
    from optimization import RandomRangeSearchCV
    from optimization import TabuSearchCV

    # Validation package
    from validation import Cross_Validation
    from validation import BootstrapCV
    from validation import KFold
    from validation import NxM
    from validation import TestCV
    from validation import LoaderCV

    # Map package
    from map import Database
    from map import DatabaseTwoClass
    from map import DatabaseNoClass
    from map import selection_db
    from map import baseline_db
    from map import validation_db
    from map import dataset_db
    from map import transformation_db
    from map import learning_db
    from map import optimization_db
    from map import metric_db
