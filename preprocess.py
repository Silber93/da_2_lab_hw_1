
import pandas as pd
from datetime import datetime
import numpy as np
# from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import pickle

TRAIN_FILENAME = 'train.tsv'
TEST_FILENAME = 'test.tsv'


COLS_FOR_MODEL = ['budget', 'original_language', 'popularity', 'runtime', 'spoken_languages', 'vote_count',
                  'production_companies', 'vote_average', 'genres', 'cast', 'crew', 'day_of_year', 'year', 'revenue']

COLS_TO_PARSE = {'genres': "'name': ",
                 'spoken_languages': "'iso_639_1': ",
                 'cast': "'name': ",
                 'crew': "'Director', 'name': ",
                 'production_companies': "'name': "}

CATEGORICAL_FEATURES = ['original_language', 'genres', 'production_companies', 'spoken_languages', 'cast', 'director']


def tsv_to_df(filepath: str):
  file_savename = filepath.replace('.tsv', '') + '.csv'
  df = pd.read_csv(filepath, sep='\t')
  return df


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
    if col == 'crew':
      df = df.drop('crew', axis=1)
      col = 'director'
    df[col] = new_col
  return df


def vec_to_features(test_preprocessed, train_preprocessed=None, train=True):
  print("vectorizing categorical features...")
  if train:
    d = {}
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
          for val in row:
            if val == '':
              continue
            if val not in d[col]:
              d[col][val] = len(d[col])
    with open('feature_dictionary.pickle', 'wb') as handle:
      pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)
    ret_df = train_preprocessed.copy()
  else:
    ret_df = test_preprocessed.copy()
    with open('feature_dictionary.pickle', 'rb') as handle:
      d = pickle.load(handle)
  for col in CATEGORICAL_FEATURES:
    if col == 'crew':
      col = 'director'
    rows = ret_df[col].values
    dummy_vals_matrix = []
    for row in rows:
      dummy_vals_row = [0] * len(d[col])
      if col == 'cast':
        row = row[:4]
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
  ret_df = mean_impute(ret_df)
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
  Y = df[['revenue']]
  X_transformed = df['budget'].mean()
  df['budget'] = [X_transformed if np.isnan(x) else x for x in df['budget'].values]
  X = df[[x for x in df.columns if x != 'revenue']].values
  Y_transformed = df['revenue'].mean()
  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(X)
  df = pd.DataFrame(X_scaled, columns=[x for x in df.columns if x != 'revenue'])
  df['revenue'] = [Y_transformed if np.isnan(x) else x for x in Y['revenue'].values]
  # df = pd.concat([df, Y], axis=1)
  return df


def run(test_filename, train_filename=None, train=True):
  print(f'running preprocess ({"train" if train else "test"})...')
  train_preprocessed = None
  test_df = tsv_to_df(test_filename)
  test_preprocessed = col_to_vec(test_df)
  if train:
    train_df = tsv_to_df(train_filename)
    train_preprocessed = col_to_vec(train_df)
  df_ready = vec_to_features(test_preprocessed, train_preprocessed, train)
  return df_ready

# df_ready = run(TEST_FILENAME, TRAIN_FILENAME, train=False)
# print(df_ready)



