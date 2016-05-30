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


def load_and_prepare_data(data_start_date, data_end_date):
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

    df_train = set_date_range(df_concat, data_start_date, data_end_date)
    return df_train


def set_date_range(df, start_date, end_date):
    li("Set date range Dates")
    ld(start_date, end_date)
    # using a 15 year perriod
    li("Using aa 15 year period of data")
    df = df[start_date:end_date].copy()
    ld(df.head())
    return df


# def add_extra_features(df, func, **kwargs):
#     if kwargs:
#         df_new = func(df, **kwargs)
#     else:
#         df_new = func(df)
#     df_X = pd.concat([df, df_new], axis=1)
#     return df_X

windows = [2, 7, 30, 90]
data_start_date = '2001-01-04'
data_end_date = '2016-01-04'
# df_train = set_date_range(df_raw, start_date, end_date)
days_ahead = [1, 7, 30, 90]


def calc_daily_ret(df):
    li("calculate daily returns")
    # Todays value divided by yesterdays
    df_dr = (df / df.shift(1)) - 1
    df_dr.columns = ["%s_%s" % (col, 'dr') for col in df.columns]
    # First row must be 0
    df_dr.iloc[0] = 0
    return df_dr


def calc_rolling_func(df_in, windows, func, prefix):
    li("caluclate rolling averages:" + prefix)
    new_columns = []
    for window in windows:
        new_df = func(df_in, window)
        new_df.columns = ["%s_%s_%s" % (col, prefix, window)
                          for col in new_df.columns]
        new_df.iloc[:window-1] = 0
        #new_df.fillna(method='bfill', inplace=True)
        #new_df.fillna(method='ffill', inplace=True)
        new_columns.append(new_df)

    result = pd.concat(new_columns, axis=1)
    return result


def calc_future_daily_ret(df_in, days_ahead=1):
    # Value from n days ahead divided by todays value
    li("Create the y labels vector")
    df_y = (df_in.shift(-days_ahead) / df_in) - 1
    # We can't speak for the future, so set the values to 0
    df_y.iloc[-days_ahead:] = 0
    return df_y


def calc_future_returns(df_in, windows):
    result = pd.DataFrame()
    for window in windows:
        ud = calc_future_daily_ret(df_in, days_ahead=window)
        result = pd.concat([result, ud], axis=1)
        result = result.rename(index=str, columns={ud.name:"%s_%s" % (ud.name, window)})
    return result


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


li('fookit, lets see is we can predict')
from sklearn.cross_validation import KFold




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
    df_train = load_and_prepare_data(data_start_date, data_end_date)

    df_dr = calc_daily_ret(df_train)
    windows = [2, 7, 30, 180]
    df_rw = calc_rolling_averages(df_dr, windows)

    df_X = pd.concat([df_dr, df_rw], axis=1)
    return df_X


def do_window_plot(df_in, window, title):
    do_plot(df_in[[col for col in df_in.columns.values if window in col]], title)


def do_plot(df_in, title, do_plot=True):
    ax = df_in.plot(figsize=(15,10))
    plt.title(title)
    if do_plot:
        plt.show()


def train(clf, X_train, y_train, param_grid=None):
    logging.debug(X_train.shape)
    logging.debug(y_train.shape)
    logging.debug(y_train.value_counts())

    if param_grid is not None:
        from sklearn.grid_search import GridSearchCV
        gs = GridSearchCV(clf, param_grid)
        gs.fit(X_train, y_train)
        clf = gs.best_estimator_
    else:
        clf.fit(X_train, y_train)

    return clf


def test(clf, X_test, y_test):
    from sklearn.metrics import f1_score
    from sklearn.metrics import classification_report
    logging.debug(clf)

    pred_actual = clf.predict(X_test)
    logging.info("Actual")
    logging.info(classification_report(y_test, pred_actual))

    logging.info("random sample")
    pred_sample = y_test.sample(y_test.shape[0])
    logging.info(classification_report(y_test, pred_sample))

    return f1_score(y_test, pred_actual)

if __name__ == '__main__':
    import numpy as np
    values = [ 0.,-0.0027907,0.00223881,-0.00316456,-0.00392157,-0.00712411,-0.0043429]
    print np.std(values, ddof=1)
    print np.mean(values)
    values = [ 0.,-0.0027907,0.00223881,-0.00316456,-0.00392157,-0.00712411]

    values = [ 0.,-0.0027907]
    print np.std(values, ddof=1)
    print np.mean(values)
