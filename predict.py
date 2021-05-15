import argparse
import numpy as np
import math
import pandas as pd
from model import *
import pickle
import preprocess


# Parsing script arguments
parser = argparse.ArgumentParser(description='Process input')
parser.add_argument('tsv_path', type=str, help='tsv file path')
args = parser.parse_args()

# Reading input TSV
# data = pd.read_csv('test.tsv', sep="\t")
data = preprocess.run(args.tsv_path, train=False)
# features = ['budget', 'popularity', 'vote_average', 'vote_count']
# X, y = split_covariates_from_target(args.tsv_path,'revenue',features)
# X, y = split_covariates_from_target('test.tsv','revenue',features)

# X_train = X_train[features].to_numpy(dtype=object)
# y_train = y_train.to_numpy(dtype=float)
# X = X[features].to_numpy(dtype=object)

infile = open('beta_hat.pkl','rb')
beta_hat = pickle.load(infile)
y_train_pred = np.dot(data, beta_hat)

# Example:
prediction_df = pd.DataFrame(columns=['id', 'revenue'])
prediction_df['id'] = data['id']
prediction_df['revenue'] = y_train_pred
####

prediction_df.to_csv("prediction.csv", index=False, header=False)

prediction = prediction_df['revenue'].values.ravel()
res = root_mean_squared_log_error(data['revenue'], prediction)
print("RMSLE is: {:.6f}".format(res))


### Utility function to calculate RMSLE
# def rmsle(y_true, y_pred):
#     """
#     Calculates Root Mean Squared Logarithmic Error between two input vectors
#     :param y_true: 1-d array, ground truth vector
#     :param y_pred: 1-d array, prediction vector
#     :return: float, RMSLE score between two input vectors
#     """
#     assert y_true.shape == y_pred.shape, \
#         ValueError("Mismatched dimensions between input vectors: {}, {}".format(y_true.shape, y_pred.shape))
#     return np.sqrt((1/len(y_true)) * np.sum(np.power(np.log(y_true + 1) - np.log(y_pred + 1), 2)))