import argparse
import numpy as np
import math
import pandas as pd
from model import *
import pickle
import preprocess
import utilities


# Parsing script arguments
parser = argparse.ArgumentParser(description='Process input')
parser.add_argument('tsv_path', type=str, help='tsv file path')
args = parser.parse_args()

# Reading input TSV
# data = pd.read_csv('test.tsv', sep="\t")
data = preprocess.run(args.tsv_path, train=False)

models = ['linear_model','ridge_model','custom_model']
for m in models:
    infile = open(f'saved_models/{m}.pkl','rb')
# result = pickle.load(infile)
    model = pickle.load(infile)
    beta_hat, b_hat = model[0],model[1]

    prediction_df = pd.DataFrame(columns=['id', 'revenue'])
    prediction_df['id'] = data['id']
    data_1 = data[[x for x in data if x != 'id']]
    X, _ = split_covariates_from_target(data_1, 'revenue')
    print("running prediction...")
    # prediction_df['revenue'] = model.predict(X)
    y_test_pred = np.dot(X, beta_hat.T)+b_hat
    prediction_df['revenue'] = y_test_pred
    ####

    prediction_df.to_csv("prediction.csv", index=False, header=False)

    prediction = prediction_df['revenue'].values.ravel()
    res = root_mean_squared_log_error(data_1['revenue'].values, prediction)
    print(f"RMSLE of {m} is: {round(res,3)}")


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