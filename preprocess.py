
import pandas as pd
from datetime import datetime
import numpy as np
# from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import pickle

TRAIN_FILEPATH = 'data/train.tsv'
TEST_FILEPATH = 'data/test.tsv'


COLS_FOR_MODEL = ['id', 'budget', 'original_language', 'popularity', 'runtime', 'vote_count',
                 'vote_average', 'genres', 'day_of_year', 'year', 'revenue']

COLS_TO_PARSE = {'genres': "'name': "}
CATEGORICAL_FEATURES = ['original_language','genres']


def col_to_vec(df):
  df['day_of_year'] = [datetime.strptime(x, '%Y-%m-%d').timetuple().tm_yday for x in df['release_date'].values]
  df['year'] = [int(x.split('-')[0]) for x in df['release_date'].values]
  df = df[COLS_FOR_MODEL]
  for col in COLS_TO_PARSE:
    col_values = df[col].values
    new_col = []
    for row in col_values:
      row = row.split(COLS_TO_PARSE[col])[1:]
      row = [x.split(',')[0].replace('\'', '').replace('}', '').replace(']', '') for x in row]
      new_col.append(tuple(row))
    df = df.drop(col, axis=1)
    if col == 'crew':
      col = 'director'
    df[col] = new_col
  return df


def vec_to_features(test_preprocessed, train_preprocessed=None, train=True):
  print("vectorizing categorical features...")
  if train:
    d = {}
    cast_dict = {}
    directors_dict = {}
    for df in [train_preprocessed, test_preprocessed]:
      for col in CATEGORICAL_FEATURES:
        if col == 'crew':
          col = 'director'
        if col not in d:
          d[col] = {}
        rows = df[col].values
        for row in rows:
          if col == 'cast':
            row = row[:2]
          if col == 'original_language':
            if row not in d[col]:
              d[col][row] = len(d[col])
            continue
          for val in row:
            if col == 'cast':
              cast_dict[val] = 1 if val not in cast_dict.keys() else cast_dict[val]+1
            if col == 'director':
              directors_dict[val] = 1 if val not in directors_dict.keys() else directors_dict[val]+1
            if val == '':
              continue
            if val not in d[col]:
              d[col][val] = len(d[col])
    for key in cast_dict.keys():
      if cast_dict[key] < 20:
        del d['cast'][key]
    for key in directors_dict.keys():
      if directors_dict[key] < 13:
        del d['director'][key]
    with open('data/feature_dictionary.pickle', 'wb') as handle:
      pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)
    ret_df = train_preprocessed.copy()
  else:
    ret_df = test_preprocessed.copy()
    with open('data/feature_dictionary.pickle', 'rb') as handle:
      d = pickle.load(handle)
  for col in CATEGORICAL_FEATURES:
    if col == 'crew':
      col = 'director'
    rows = ret_df[col].values
    dummy_vals_matrix = []
    for row in rows:
      dummy_vals_row = [0] * len(d[col])
      if col == 'cast':
        row = row[:2]
      for val in row:
        if val == '':
          continue
        try:
          dummy_vals_row[d[col][val]] = 1
        except:
          pass
      dummy_vals_matrix.append(dummy_vals_row)
    col_names = [col + '_' + item[0] for item in sorted(d[col].items(), key=lambda x: x[1])]
    dummy_df = pd.DataFrame(data=dummy_vals_matrix, columns=col_names)
    ret_df = pd.concat([ret_df, dummy_df], axis=1)
    ret_df = ret_df.drop(col, axis=1)
  return ret_df


# def knn_impute(df):
#   print('performing imputation...')
#   df['budget'] = [np.nan if x <= 100 else x for x in df['budget'].values]
#   df['revenue'] = [np.nan if x <= 100 else x for x in df['revenue'].values]
#   imputer = KNNImputer(n_neighbors=1)
#   X = df[[x for x in df.columns if x != 'revenue']].values
#   Y = df[['revenue']]
#   scaler = StandardScaler()
#   X_scaled = scaler.fit_transform(X)
#   df = pd.DataFrame(X_scaled, columns=[x for x in df.columns if x != 'revenue'])
#   X_transformed = imputer.fit_transform(df[['budget']].values)
#   df['budget'] = X_transformed
#   df = pd.concat([df, Y], axis=1)
#   df['budget'] = X_transformed
#   Y_transformed = imputer.fit_transform(df[['revenue']].values)
#   df['revenue'] = Y_transformed
#
#   return df


def mean_impute(df):
  print('performing imputation...')
  df['budget'] = [np.nan if x <= 100 else x for x in df['budget'].values]
  df['revenue'] = [np.nan if x <= 100 else x for x in df['revenue'].values]
  mean_budget = df['budget'].mean()
  df['budget'] = [mean_budget if np.isnan(x) else x for x in df['budget'].values]
  mean_revenue = df['revenue'].mean()
  df['revenue'] = [mean_revenue if np.isnan(x) else x for x in df['revenue'].values]
  return df


def scale(df):
  scaler = StandardScaler()
  X = df[[x for x in df.columns if x not in ['id', 'revenue']]].values
  X_scaled = scaler.fit_transform(X)
  Y = df[['revenue']]
  id_df = df[['id']]
  df = pd.DataFrame(X_scaled, columns=[x for x in df.columns if x not in ['id', 'revenue']])
  df = pd.concat([id_df, df, Y], axis=1)
  return df


def run(test_filename=None, train_filename=None, train=True):
  print("\t----PREPROCESS----")
  test_filename = TEST_FILEPATH if test_filename is None else test_filename
  train_filename = TRAIN_FILEPATH if train_filename is None else train_filename
  print(f'running preprocess ({"train" if train else "test"})...')
  train_preprocessed = None
  test_df = pd.read_csv(test_filename, sep='\t')
  test_preprocessed = col_to_vec(test_df)
  if train:
    train_df = pd.read_csv(train_filename, sep='\t')
    train_preprocessed = col_to_vec(train_df)
  df_ready = vec_to_features(test_preprocessed, train_preprocessed, train)
  df_ready = mean_impute(df_ready)
  df_ready = scale(df_ready)
  df_ready.dropna(inplace=True, how='any')
  # df_ready.dropna(inplace=True, subset=[target], how='any')
  print(f"preprocess completed, frame shape: {df_ready.shape}\n")
  return df_ready



