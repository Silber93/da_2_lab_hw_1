import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_log_error
from sklearn import linear_model
import numpy as np
import math
from scipy.optimize import minimize
import pickle


def run_linear_reg(X_train, y_train):
    print("running linear regression...")
    linreg = LinearRegression(normalize=True)
    lr_model = linreg.fit(X_train, y_train)
    beta_hat = lr_model.coef_
    b = lr_model.intercept_
    pickle.dump([beta_hat, b], open("saved_models/linear_model.pkl", "wb"))
    y_train_pred = lr_model.predict(X_train)
    y_train_pred[y_train_pred < 0] = 0
    train_score = mean_squared_log_error(y_train, y_train_pred)
    train_score = np.sqrt(train_score)
    print("linear regression RMSLE on the train set is:")
    print(train_score)
    print("linear model saved in saved models/linear_model.pkl\n")
    return train_score


def run_ridge_reg(X_train, y_train):
    print("running ridge regression...")
    reg = linear_model.RidgeCV(alphas=np.logspace(-6, 6, 13))
    reg.fit(X_train, y_train)
    beta_hat = reg.coef_
    b = reg.intercept_
    pickle.dump([beta_hat, b], open("saved_models/ridge_model.pkl", "wb"))
    y_train_pred = reg.predict(X_train)
    y_train_pred[y_train_pred < 0] = 0
    train_score = mean_squared_log_error(y_train, y_train_pred)
    train_score = np.sqrt(train_score)
    print("ridge regression RMSLE on the train set is:")
    print(train_score)
    print("ridge model saved in saved models/ridge_model.pkl\n")
    return train_score


def run_custom_reg(X_train, y_train):
    print("running custom regression...")
    beta_init = np.ones(X_train.shape[1])
    result = minimize(objective_function, beta_init, args=(X_train, y_train),
                      method='BFGS', options={'maxiter': 500})
    beta_hat = result.x
    print("beta hat = ", beta_hat)
    y_train_pred = np.dot(X_train, beta_hat)
    y_train_pred[y_train_pred < 0] = 0
    # pred_diff = y_train_pred - y_train
    train_score = root_mean_squared_log_error(y_train,y_train_pred)
    print("custom regression score on the train set is:")
    print(train_score)
    pickle.dump(beta_hat, open("saved_models/custom_model.pkl", "wb"))
    print("custom model saved in saved models/custom_model.pkl\n")
    return train_score


def objective_function(beta, X, y):
    loss_function = root_mean_squared_log_error
    # loss_function = rmsle
    y_hat = np.dot(X, beta)
    error = loss_function(y_hat, y)
    return (error)


def root_mean_squared_log_error(y_true, y_pred):
    try:
        y_pred = np.array([x[0] for x in y_pred])
    except:
        pass
    y_pred = np.array([y if y>0 else 0 for y in y_pred])
    y_true = np.array([y if y>0 else 0 for y in y_true])
    N = len(y_true)
    a = np.array([math.log(y+1) for y in y_true])
    b = np.array([math.log(y+1) for y in y_pred])
    return (np.sqrt(np.sum(np.power((a-b),2))/N))


def rmsle(y_true, y_pred):
    """
    Calculates Root Mean Squared Logarithmic Error between two input vectors
    :param y_true: 1-d array, ground truth vector
    :param y_pred: 1-d array, prediction vector
    :return: float, RMSLE score between two input vectors
    """
    assert y_true.shape == y_pred.shape, \
        ValueError("Mismatched dimensions between input vectors: {}, {}".format(y_true.shape, y_pred.shape))
    return np.sqrt((1/len(y_true)) * np.sum(np.power(np.log(y_true + 1) - np.log(y_pred + 1), 2)))