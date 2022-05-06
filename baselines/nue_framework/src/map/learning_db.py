from map import DatabaseTwoClass
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.linear_model import LinearRegression, Ridge, RidgeClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import BaggingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.dummy import DummyRegressor, DummyClassifier
from learning import Learner
from learning import MLPReg


knn_class = [KNeighborsClassifier, KNeighborsRegressor]
lr_class = [None, LinearRegression]
gpr_class = [None, GaussianProcessRegressor]
rr_class = [RidgeClassifier, Ridge]
svm_class = [SVC, SVR]
cart_class = [DecisionTreeClassifier, DecisionTreeRegressor]
rf_class = [RandomForestClassifier, RandomForestRegressor]

# Order: Classification, Regression
learning_db = DatabaseTwoClass(Learner,
                  { 
                    "dummy":[DummyClassifier, DummyRegressor],
                    "knn": knn_class,
                    # Left out for now
                    # "mlp":MLPReg,
                    # "multilayerperceptron":MLPReg,
                    # "mlp-sigmoid":MLPReg,
                    # "mlp-tanh":MLPReg,
                    # "mlp-relu":MLPReg,
                    "lr":lr_class,
                    "ols":lr_class,
                    "linearregression":lr_class,
                    "gp":gpr_class,
                    "gaussianprocesses":gpr_class,
                    "ridgeregression":rr_class,
                    "ridge":rr_class,
                    "svr":svm_class,
                    "svc":svm_class,
                    "svm":svm_class,
                    # Removed for now
                    # "bagging":BaggingRegressor,
                    "decisionstump":cart_class,
                    "reptree":cart_class,
                    "regressiontree":cart_class,
                    "cart":cart_class,
                    "rf":rf_class,
                    "randomforest":rf_class,
                    },
                  {"decisionstump":{"max_depth",1} ,
                  #  "bagging":{"base_estimator",DecisionTreeRegressor(max_depth = 1)},
                  #  "mlp-sigmoid":{"activation":"logistic"},
                  #  "mlp-tanh":{"activation":"tanh"},
                  #  "mlp-relu":{"activation":"relu"},
                   })
