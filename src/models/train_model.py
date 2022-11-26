import numpy as np
import time
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

def build_model(model_name: str, parameters: dict):
    """
    :param model_name: The name of the model
    :param parameters: The parameters of the model
    :return: A model instantiated with the correct parameters
    """
    if model_name == "LinearRegressor":
        return LinearRegression()
    elif model_name == "SVMRegressor":
        return SVR(kernel=parameters['kernel'], degree=parameters['degree'], gamma=parameters['gamma'],
                   coef0=parameters['coef0'], tol=parameters['tol'], C=parameters['C'], epsilon=parameters['epsilon'])
    elif model_name == "SGDRegressor":
        return SGDRegressor(loss=parameters['loss'])
    elif model_name == "RandomForestRegressor":
        return RandomForestRegressor(n_estimators=parameters['n_estimators'], max_depth=parameters['max_depth'])
    elif model_name == "AdaBoostRegressor":
        return AdaBoostRegressor(n_estimators=parameters['n_estimators'], learning_rate=parameters['learning_rate'])
    elif model_name == "MLPRegressor":
        return MLPRegressor(hidden_layer_sizes=parameters['hidden_layer_sizes'], activation=parameters['activation'],
                            solver=parameters['solver'], learning_rate=parameters['learning_rate'],
                            max_iter=parameters['max_iter'])


def train_model(model_name: str, parameters: dict, X: np.ndarray, y: np.ndarray, n_splits: int):
    """
    :param model_name: name of the model.
    :param parameters: dictionary of the parameters
    :param X: training matrix
    :param y: target vector
    :param n_splits: number of folds for the K-fold training strategy
    :return: The model with the correct parameters trained using cross validation on the given data.
    """
    model = build_model(model_name, parameters)

    # K-Fold training strategy
    kf = KFold(n_splits)
    error = 0
    start = time.time()
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model.fit(X_train, y_train)
        # run predictions
        y_pred = model.predict(X_test)
        error += mean_squared_error(y_test, y_pred)
    stop = time.time()

    print("Training time: ", stop - start, "      mean MSE:  ", error / n_splits)
    return model
