from utilities import *
import numpy as np
from scipy.optimize import minimize
import pickle
import preprocess

def tsv_to_df(filepath: str):
  file_savename = filepath.replace('.tsv', '') + '.csv'
  print(file_savename)
  df = pd.read_csv(filepath, sep='\t')
  df.to_csv(file_savename, index=False)
  return df


def split_covariates_from_target(df,target,features):
    # df = tsv_to_df(filepath=filepath)
    print(df)
    df.dropna(inplace=True,subset=features,how='any')
    df.dropna(inplace=True,subset=[target],how='any')
    print(df)
    # train_df = train_df.apply(lambda x: pd.to_numeric(x) if x.name in features else x)
    # df = df[pd.notnull(df[target])]
    # df = df[pd.notnull(df[features])]
    X = df.loc[:, df.columns != target]
    y = df.loc[:, df.columns == target]
    return X,y


def run_models(X_train,y_train):
    errors = {}
    errors['linear'] = run_linear_reg(X_train, y_train)
    errors['ridge'] = run_ridge_reg(X_train, y_train)
    errors['custom'] = run_custom_reg(X_train, y_train)


def train_model():
    df = preprocess.run("test.tsv","train.tsv",train=True)
    features = [col for col in df.columns if col!='revenue']

    X_train, y_train = split_covariates_from_target(df,'revenue',features)
    # X_test, y_test = split_covariates_from_target(df,'revenue',features)

    X_train = X_train[features].to_numpy(dtype=object)
    y_train = y_train.to_numpy(dtype=float)
    print("printing data")
    print(X_train)
    print(y_train)
    # X_test = X_test[features].to_numpy(dtype=object)
    # y_test = y_test.to_numpy(dtype=float)

    run_models(X_train,y_train)

    beta_init = np.ones(X_train.shape[1])
    result = minimize(objective_function, beta_init, args=(X_train, y_train),
                      method='BFGS', options={'maxiter': 500})

    beta_hat = result.x
    pickle.dump(beta_hat, open("beta_hat.pkl", "wb"))
    print("beta hat = ", beta_hat)

if __name__ == '__main__':
    train_model()


    # y_hat = np.dot(X_test,beta_hat)
    # pred_diff =y_hat-y_test
    # error = root_mean_squared_log_error(y_hat, y_test)
    # print("RMSLE on the test set is =", error)

    # print("pred_diffpred_diff)
    # print(y_hat.shape)
    # print(y_test.shape)
    # df = pd.DataFrame(data=zip(y_test,y_hat,pred_diff), columns=['real','prediction','error'])
    # print(df)

    # print(train_df.shape)
    # print(train_df.columns)
    # print(test_df.shape)
    # print(test_df.columns)

    # run_linear_reg(X_train,y_train,X_test,y_test,features)
