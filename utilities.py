import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from sklearn import linear_model
import numpy as np
import math
from scipy.optimize import minimize


def run_linear_reg(X_train, y_train):
    linreg = LinearRegression(normalize=True)
    lr_model = linreg.fit(X_train, y_train)
    y_train_pred = lr_model.predict(X_train)
    y_train_pred[y_train_pred < 0] = 0
    train_score = mean_squared_log_error(y_train, y_train_pred)
    train_score = np.sqrt(train_score)
    print("linear regression RMSLE on the train set is:")
    # print(lr_model.score(X_train[features], y_train.values.ravel()))
    print(train_score)
    return train_score


def run_ridge_reg(X_train, y_train):
    reg = linear_model.RidgeCV(alphas=np.logspace(-6, 6, 13))
    reg.fit(X_train,y_train)
    y_train_pred = reg.predict(X_train)
    y_train_pred[y_train_pred < 0] = 0
    train_score = mean_squared_log_error(y_train, y_train_pred)
    train_score = np.sqrt(train_score)
    print("ridge regression RMSLE on the train set is:")
    # print(lr_model.score(X_train[features], y_train.values.ravel()))
    print(train_score)
    return train_score


def run_custom_reg(X_train, y_train):
    beta_init = np.ones(X_train.shape[1])
    # y_test = np.array([x[0] for x in y_test])
    result = minimize(objective_function, beta_init, args=(X_train, y_train),
                      method='BFGS', options={'maxiter': 500})
    beta_hat = result.x
    print("beta hat = ", beta_hat)
    y_train_pred = np.dot(X_train, beta_hat)
    y_train_pred[y_train_pred < 0] = 0
    # pred_diff = y_train_pred - y_train
    train_score = root_mean_squared_log_error(y_train_pred, y_train)
    print("custom regression score on the train set is:")
    print(train_score)
    return train_score


def objective_function(beta, X, y):
    loss_function = root_mean_squared_log_error
    # loss_function = rmsle
    y_hat = np.dot(X, beta)
    error = loss_function(y_hat, y)
    return (error)


def root_mean_squared_log_error(y_pred, y_true):
    try:
        y_true = np.array([x[0] for x in y_true])
    except:
        pass
    y_pred = np.array([y if y>0 else 0 for y in y_pred])
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

#
# X_train = X_train[features].to_numpy(dtype=object)
#     y_train = y_train.to_numpy(dtype=float)
#     X_test = X_test[features].to_numpy(dtype=object)
#     y_test = y_test.to_numpy(dtype=float)
#
#     # print(X_train)
#
#     loss_function = root_mean_squared_log_error
#
#     beta_init = np.ones(X_train.shape[1])
#     result = minimize(objective_function, beta_init, args=(X_train, y_train),
#                       method='BFGS', options={'maxiter': 500})
#
#     # The optimal values for the input parameters are stored
#     # in result.x
#     beta_hat = result.x
#     print("beta hat = ", beta_hat)
#     # print(beta_hat.shape)
#     # print(X_test.shape)
#     # print(type(beta_hat))
#     y_hat = np.dot(X_test,beta_hat)
#     y_true = np.array([x[0] for x in y_test])
#     pred_diff =y_hat-y_true
#     sum = 0
#     for true,pred in zip(y_true,y_hat) :
#         # print(diff)
#         # np.sqrt(sum((np.log(y_true + 1) - np.log(y_pred + 1)) ** 2) / N))
#         sum += (np.log(true+1) - np.log(pred+1))** 2
#     print(sum)
#     sum/=y_hat.shape[0]
#     print(sum)
#     res = np.sqrt(sum)
#
#     # print("pred_diff = ",pred_diff)
#     print(y_hat.shape)
#     print(y_true.shape)
#     df = pd.DataFrame(data=zip(y_true,y_hat,pred_diff), columns=['real','prediction','error'])
#     print(df)
#     error = root_mean_squared_log_error(y_hat,y_test)
#     print("RMSLE on the test set is = ", error)
#
#     # run_linear_reg(X_train,y_train,X_test,y_test,features)
