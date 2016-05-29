import pandas as pd
import re
import quandl
import os

import matplotlib.pyplot as plt
import datetime
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.info("LOADING Helper1 :-)" + str(datetime.datetime.today()))


li = logging.info
ld = logging.debug


def download_quandl(codes, filename=None, load_file=True):
    if filename is None:
        filename = re.sub('[^-a-zA-Z0-9_.() ]+', '_', codes)

    if load_file and os.path.exists(filename):
        li("Loading file:%s" % filename)
        return pd.read_csv(filename)

    the_data = quandl.get(codes)
    the_data.describe()
    the_data.head()
    the_data.to_csv(filename)
    return the_data

currencies = {
    'AUD': 'FRED/DEXUSAL',
    'JPY': 'FRED/DEXJPUS',
    'GBP': 'FRED/DEXUSUK',
    'EUR': 'FRED/DEXUSEU',
    'CAD': 'FRED/DEXCAUS',
    'CHF': 'FRED/DEXSZUS',
    'CNY': 'FRED/DEXCHUS',
    'NZD': 'FRED/DEXUSNZ',
}


def load_and_prepare_data():
    li("Load currencies")
    df_curr = download_quandl([currencies[k]
                               for k in currencies], 'currencies')
    ld(df_curr.head())
    # df_curr.describe()
    df_curr.set_index('DATE', inplace=True)
    df_curr.columns = [a for a in currencies]

    li("Are we using the correct timezone?")
    # FRED = noon NYC time
    # London 10:30,3pm
    # NYC

    li("inverse currencies so they are all 'how many x does 1 usd buy'")
    for curr in currencies:
        if currencies[curr][:-2] != 'US':
            df_curr[curr] = 1. / df_curr[curr]
    ld(df_curr['GBP'][:10])

    li("Lets get the gold")
    df_gold = download_quandl("LBMA/GOLD")
    ld(df_gold.head())
    df_gold.set_index('Date', inplace=True)
    pd.concat([df_gold, df_curr], axis=1)
    df_gold.drop([c for c in df_gold.columns.values if c !=
                  'USD (AM)'], axis=1, inplace=True)
    df_gold.columns = ['GOLD']

    df_concat = pd.concat([df_gold, df_curr], axis=1)

    li("Forward fill weekends and holidays")
    ld(df_concat.isnull().sum())
    df_concat.fillna(method='ffill', inplace=True)
    ld(df_concat.isnull().sum())
    return df_concat


def set_date_range(df, start_date, end_date):
    li("Set date range Dates")
    ld(start_date, end_date)
    # using a 15 year perriod
    li("Using aa 15 year period of data")
    df = df[start_date:end_date].copy()
    ld(df.head())
    return df

# df_raw = load_and_prepare_data()

# start_date = '2001-01-04'
# end_date = '2016-01-04'
# df_train = set_date_range(df_raw, start_date, end_date)


def calc_daily_ret(df):
    li("calculate daily returns")
    # calculate dailt returns
    df_dr = (df / df.shift(1)) - 1
    df_dr.columns = ["%s_%s" % (col, 'dr') for col in df.columns]
    df_dr.fillna(method='bfill', inplace=True)
    return df_dr


def calc_rolling_averages(df_in, windows):
    li("caluclate rolling averages")
    # caluclate rolling averages

    def calc_rolling_mean(df_in, windows):
        new_columns = []
        for window in windows:
            new_df = pd.rolling_mean(df_in, window)
            new_df.columns = ["%s_%s" % (col, window)
                              for col in new_df.columns]
            new_df.fillna(method='bfill', inplace=True)
            new_columns.append(new_df)

        result = pd.concat(new_columns, axis=1)
        return result

    # Get rolling averages from daily returns
    df_dr_rm = calc_rolling_mean(df_in, windows)
    return df_dr_rm


def draw_info(df_in, windows):
    def draw_graphs(df_in, cols, windows=None):
        if windows:
            cols = ['%s_%s' % (col, window)
                    for col in cols for window in windows]

        ld(cols)
        df_in[cols].plot(figsize=(15, 5)).legend(
            loc='center left', bbox_to_anchor=(1, 0.5))
        plt.axhline(0)
        plt.show()

    li("Draw a graph")
    draw_graphs(df_dr_rm, df_dr.columns, [180])
    #draw_graphs(df_dr_rm, df_dr.columns, [30])

    #df_all = pd.concat([df, df_dr, df_dr_rm], axis=1)

# df_dr = calc_daily_ret(df_train)
# windows = [2, 7, 30, 180]
# df_rw = calc_rolling_averages(df_dr, windows)

# df_y = create_y_labels(df_dr)

li('fookit, lets see is we can predict')
from sklearn.cross_validation import KFold

from sklearn.grid_search import GridSearchCV


def test_train(X_in, y_in, clf, param_grid, columns=None):
    ld(X_in.shape)
    ld(y_in.shape)
    ld(X_in.columns.values)

    li("Ensure no nulls in dataset")
    if X_in.isnull().any().any():
        raise "Empy values in dataset"

    if columns:
        ld(columns)
        X_in = X_in[columns]
    ld(X_in.columns.values)

    li('split the data')
    #X_train, X_test, y_train, y_test = train_test_split(X_in, y_in, test_size=0.3)

    gs = GridSearchCV(clf, param_grid)

    ld("Training")
    gs.fit(X_in, y_in)

    ld("Trained")
    #return gs.best_estimator_
    return gs


def load_and_calculate():
    df_raw = load_and_prepare_data()
    start_date = '2001-01-04'
    end_date = '2016-01-04'
    df_train = set_date_range(df_raw, start_date, end_date)
    df_dr = calc_daily_ret(df_train)
    windows = [2, 7, 30, 180]
    df_rw = calc_rolling_averages(df_dr, windows)

    df_X = pd.concat([df_dr, df_rw], axis=1)
    return df_X


def create_y_labels(df_in, days_ahead=1, threshold=0.):
    # Extract the labels vector
    li("Create the y labels vector")
    df_y = (df_in.shift(-days_ahead) > threshold) * 1. + 1
    return df_y
