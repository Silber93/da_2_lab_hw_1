import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
# from preproccess import COLS_TO_PARSE
from math import log


CORRELATION_COLUMNS = ['budget', 'runtime', 'vote_average', 'vote_count', 'popularity',
                       ]

CONT_FEATURES = ['budget', 'revenue', 'vote_count', 'vote_average', 'runtime', 'popularity']


CATEGORICAL_FEATURES = ['original_language', 'genres', 'production_companies', 'spoken_languages', 'cast', 'director']


CSV_TRAIN = 'train.csv'
CSV_TEST = 'test.csv'

PREPROCESSED_TRAIN = 'train_preprocessed.csv'
PREPROCESSED_TEST = 'test_preprocessed.csv'

def missing_values():
  df = pd.concat([pd.read_csv('train.csv'), pd.read_csv('test.csv')])
  print(f'all data: {df.shape[0]}')
  print(f'missing budget: {df[df["budget"] <= 1000].shape[0]}')
  print(f'missing revenue: {df[df["revenue"] <= 1000].shape[0]}')
  for col in df:
    print(col, len([1 for x in df[col].values if x in ['', np.NAN, np.nan]]), df[df[col].isna()].shape[0])
  print()
  for col in CATEGORICAL_FEATURES:
    col = 'crew' if col == 'director' else col
    print(col, len([1 for x in df[col].values if x in ['[]', '()']]))


def col_to_vec_analyze(df, col):
  df = df[df['budget'] > 0]
  hist = {}
  x = []
  y = []
  rows = df[[col, 'revenue']].values
  for row in rows:
    vals = row[0].replace('\'', '').replace('}', '').replace(']', '').replace('(', '').replace(')', '').split(',')
    for v in vals:
      if v == '':
        continue
      x.append(v)
      y.append(row[1])
      if v not in hist:
        hist[v] = []
      hist[v].append(row[1])
  hist = {i: sum(hist[i])/len(hist[i]) for i in hist}
  hist = hist.items()
  hist = sorted(hist, key=lambda x: x[1], reverse=False)
  return hist


def analyze_day(df):
  df = df[df['budget'] > 0]
  hist = {}
  for v in df[['day_of_year', 'revenue']].values:
    if v[0] not in hist:
      hist[v[0]] = []
    hist[v[0]].append(int(v[1]))
  hist = {i: sum(hist[i])/len(hist[i]) for i in hist}
  return hist


def plot_correlations():
  df = pd.concat([pd.read_csv(x) for x in ['train_preprocessed.csv', 'test_preprocessed.csv']])
  df = df[df['budget'] > 0]
  rev = df['revenue'].values
  fig = plt.figure(figsize=(20, 10))
  gs = GridSpec(2, 5)  # 2 rows, 3 columns
  axes = []
  positions = [(1, 2), (0, 3), (1, 3), (0, 4), (1, 4)]
  genres_hist = col_to_vec_analyze(df, 'genres')
  axes.append(fig.add_subplot(gs[:, 0]))
  axes[-1].barh([x[0] for x in genres_hist], [x[1] for x in genres_hist], color='g')
  axes[-1].set_title('revenue by genres')
  axes[-1].set_xlabel('average revenue')
  language_hist = col_to_vec_analyze(df, 'original_language')
  axes.append(fig.add_subplot(gs[:, 1]))
  axes[-1].barh([x[0] for x in language_hist], [x[1] for x in language_hist], color='g')
  axes[-1].set_title('revenue by languages')
  axes[-1].set_xlabel('average revenue')
  day_hist = analyze_day(df)
  axes.append(fig.add_subplot(gs[0, 2]))
  axes[-1].bar(day_hist.keys(), day_hist.values())
  axes[-1].set_title(f'day_of_year <-> revenue')
  for i, col in enumerate(CORRELATION_COLUMNS):
    x = df[col].values
    axes.append(fig.add_subplot(gs[positions[i][0], positions[i][1]]))
    axes[-1].scatter(x, rev)
    axes[-1].set_title(f'{col} <-> revenue')
  fig.suptitle('correlations with revenue', fontsize=20)
  fig.text(0.92, 0.5, 'revenue', va='center', rotation='vertical', fontsize=14)
  plt.show()


def feature_distribution():
    df = pd.concat([pd.read_csv(PREPROCESSED_TRAIN), pd.read_csv(PREPROCESSED_TEST)])
    d = df.describe()
    d.to_csv('described_numeric_data.csv')
    for col in df:
      if col in d.columns:
        continue
      print(col, len(df[col].unique()))


def cont_features_hist():
  df = pd.concat([pd.read_csv(PREPROCESSED_TRAIN), pd.read_csv(PREPROCESSED_TEST)])
  fig, axs = plt.subplots(2, 3)
  fig.set_figwidth(20)
  fig.set_figheight(10)
  fig.suptitle("continuous features distributions", fontsize=20)
  for i, col in enumerate(CONT_FEATURES):
    tmp = df[col].values
    # create a horizontal plot

    axs[int(i/3), i % 3].hist(tmp, bins=182)
    axs[int(i / 3), i % 3].title.set_text(col)
    # axs.title(f'{col}_histogram')
  plt.show()


def categorical_feature_distribution():
  df = pd.concat([pd.read_csv(PREPROCESSED_TRAIN), pd.read_csv(PREPROCESSED_TEST)])
  unique_vals = []
  fig, axs = plt.subplots(2, 3)
  fig.set_figwidth(22)
  fig.set_figheight(10)
  fig.suptitle("categorical features distributions", fontsize=20)
  for i, col in enumerate(CATEGORICAL_FEATURES):
    hist = {}
    vals = df[col].values
    for row in vals:
      row = row.replace('(', '').replace(')', '').replace('\'', '').replace(' ', '').split(',')
      for v in row:
        if v != '':
          hist[v] = 1 if v not in hist else hist[v] + 1
    plt.figure(figsize=(20, 10))
    if col in ['director', 'cast', 'production_companies', 'spoken_languages']:
      count_hist = {}
      count_list = []
      for v in hist.values():
        count_hist[v] = 1 if v not in count_hist else count_hist[v] + 1
        v = log(v) if col == 'production_companies' else v
        count_list.append(v)
      print(sorted(hist.items(), key=lambda x: x[1], reverse=True)[:10])
      # plt.bar(count_hist.keys(), count_hist.values())
      # plt.hist(count_list, bins=20)
      axs[int(i / 3), i % 3].hist(count_list, bins=20)
      axs[int(i / 3), i % 3].title.set_text(col)
      axs[int(i / 3), i % 3].set_xlabel('number of movies' + ('(log count)' if col == 'production_companies' else ''))
    else:
      hist = sorted(hist.items(), key=lambda x: x[1], reverse=False)
      # plt.barh([x[0] for x in hist], [x[1] for x in hist])
      axs[int(i / 3), i % 3].barh([x[0] for x in hist], [x[1] for x in hist])
      axs[int(i / 3), i % 3].tick_params(axis='y', which='major', labelsize=8)
      axs[int(i / 3), i % 3].title.set_text(col)
    unique_vals.append([col, len(hist)])
  for d in unique_vals:
    print(d[0], d[1])
  plt.show()
  # plt.savefig('cat_dist.png')



missing_values()
feature_distribution()
cont_features_hist()
categorical_feature_distribution()
plot_correlations()
