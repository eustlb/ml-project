import numpy as np
import time
import pandas as pd
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV, ShuffleSplit

default_parameters = {
    'LinearRegressor': {},
    'SVMRegressor': {'kernel': ['linear', 'rbf', 'sigmoid', 'poly'],
                    'C'     :[1, 10], 
                    'degree': [2, 3],
                    'gamma' : ['scale', 'auto']
                    },
    'SGDRegressor' : {'penalty':['l1', 'l2'],
                    'fit_intercept' : [True,False],
                    },
    'RandomForestRegressor' : {'n_estimators':[10,20,50,70,100], 
                                'max_depth' : [2,5,10,None],
                                },       
    'AdaBoostRegressor' : {'n_estimators':[50, 100, 150], 
                            'learning_rate' : [0.1,0.5,1]
                            },
    'MLPRegressor' : {'hidden_layer_sizes' : [(64,64), (128,128), (256,256), (128,256)],
                    'activation' : ['identity', 'logistic', 'tanh', 'relu']}                  }

def build_model(model_name: str):
    """
    :param model_name: The name of the model
    :return: The correct model instanciated
    """
    if model_name == "LinearRegressor":
        return LinearRegression()
    elif model_name == "SVMRegressor":
        return SVR()
    elif model_name == "SGDRegressor":
        return SGDRegressor()
    elif model_name == "RandomForestRegressor":
        return RandomForestRegressor()
    elif model_name == "AdaBoostRegressor":
        return AdaBoostRegressor()
    elif model_name == "MLPRegressor":
        return MLPRegressor()

def train_model(model_name: str, parameters: dict, X: np.ndarray, y: np.ndarray, n_splits: int):
    """
    This function instantiates a model with the correct specified parameters and trains it on the data using cross
    validation with the given number of splits. Parameters should be specified in a grid-like format. If more than one
    combination is possible, a grids earch is done and the most optimal model is returned. If no parameters are provided,
    we do a grid search with default parameters grid and return the most optimal model.
    :param model_name: The name of the model to be used.
    Available models are : ['LinearRegressor' , 'SVMRegressor', 'SGDRegressor' , 'RandomForestRegressor', 'AdaBoostRegressor', 'MLPRegressor']
    :param parameters: Parameters of the model. They should be specified in a grid-like format.
    Example : {'kernel':['linear', 'rbf', 'sigmoid', 'poly'],
                    'C'     :[1, 10],
                    'degree': [2, 3],
                    'gamma' : ['scale', 'auto']
                    }
    :param X: Training data
    :param y: Training labels
    :param n_splits: Number of splits for cross validation
    :return: A model with the most optimal parameters from the given combination trained on the given data
    """

    if parameters is None or parameters == {}:
        parameters = default_parameters[model_name]

    model = build_model(model_name)
    cvp = ShuffleSplit(n_splits=n_splits)
    clf = GridSearchCV(
        model, parameters, scoring='neg_mean_squared_error', cv=cvp)

    start = time.time()
    clf.fit(X, y)
    stop = time.time()

    print("Training time: ", stop - start, "      best score:  ", -
          clf.best_score_, "    best params: ", clf.best_params_)
    return clf.best_estimator_
