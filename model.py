from utilities import *
import numpy as np
from scipy.optimize import minimize
import pickle
import preprocess
from sklearn.model_selection import KFold


def split_covariates_from_target(df,target):
    X = df.loc[:, df.columns != target]
    y = df.loc[:, df.columns == target]
    return X,y


def run_models(X_train,y_train):
    run_linear_reg(X_train, y_train)
    run_ridge_reg(X_train, y_train)
    run_custom_reg(X_train, y_train)


def ten_fold_cv(method):
    df = preprocess.run(train=True)
    kf = KFold(n_splits=10)
    df = df[[x for x in df if x != 'id']]
    features = [col for col in df.columns if col != 'revenue']
    X, y = split_covariates_from_target(df, 'revenue')
    X = X[features].to_numpy(dtype=object)
    y = y.to_numpy(dtype=float)
    kf.get_n_splits(X)
    errors = []
    for train_index, test_index in kf.split(X):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        beta_hat = apply_custom_reg(X_train, y_train,method)
        y_test_pred = np.dot(X_test,beta_hat.T)
        errors.append(root_mean_squared_log_error(y_test,y_test_pred))
    return sum(errors)/10


def apply_custom_reg(X_train, y_train, method):
    print("running custom regression...")
    beta_init = np.ones(X_train.shape[1])
    # y_test = np.array([x[0] for x in y_test])
    result = minimize(objective_function, beta_init, args=(X_train, y_train),
                      method=method, options={'maxiter': 500})
    beta_hat = result.x
    y_train_pred = np.dot(X_train, beta_hat)
    y_train_pred[y_train_pred < 0] = 0
    # pred_diff = y_train_pred - y_train
    train_score = root_mean_squared_log_error(y_train,y_train_pred)
    print("custom regression score on the train set is:")
    print(train_score)
    pickle.dump(beta_hat, open("saved_models/custom_model.pkl", "wb"))
    print("custom model saved in saved models/custom_model.pkl\n")
    return beta_hat


def train_model(X_train,y_train):
    print("\t----MODEL----")
    run_models(X_train,y_train)
    # ten_fold_cv(method)


if __name__ == '__main__':
    df = preprocess.run(train=True)
    df = df[[x for x in df if x != 'id']]
    features = [col for col in df.columns if col != 'revenue']
    X, y = split_covariates_from_target(df, 'revenue')
    X = X[features].to_numpy(dtype=object)
    y = y.to_numpy(dtype=float)
    train_model(X,y)

    # --------------------- Cross-Validation --------------------- #
    # for method in ['BFGS','Nelder-Mead','Powell','CG','Newton-CG']:
    #     result = ten_fold_cv(method)
    #     print(f"{method} res = {result}")
