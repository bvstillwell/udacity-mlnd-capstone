import pandas as pd
import re
import quandl
import os

import matplotlib.pyplot as plt

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.info("test")

# Log message


def lm(text):
    logging.info(text)
    logging.info(text)


# import quandl
def download_quandl(codes, filename=None, load_file=True):
    if filename is None:
        filename = re.sub('[^-a-zA-Z0-9_.() ]+', '_', codes)

    if load_file and os.path.exists(filename):
        lm("Loading file:%s" % filename)
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
    lm("Load currencies")
    df_curr = download_quandl([currencies[k]
                               for k in currencies], 'currencies')
    logging.debug(df_curr.head())
    # df_curr.describe()
    df_curr.set_index('DATE', inplace=True)
    df_curr.columns = [a for a in currencies]

    lm("Are we using the correct timezone?")
    # FRED = noon NYC time
    # London 10:30,3pm
    # NYC

    lm("inverse currencies so they are all 'how many x does 1 usd buy'")
    for curr in currencies:
        if currencies[curr][:-2] != 'US':
            df_curr[curr] = 1. / df_curr[curr]
    logging.debug(df_curr['GBP'][:10])

    lm("Lets get the gold")
    df_gold = download_quandl("LBMA/GOLD")
    logging.debug(df_gold.head())
    df_gold.set_index('Date', inplace=True)
    pd.concat([df_gold, df_curr], axis=1)
    df_gold.drop([c for c in df_gold.columns.values if c !=
                  'USD (AM)'], axis=1, inplace=True)
    df_gold.columns = ['GOLD']

    df_concat = pd.concat([df_gold, df_curr], axis=1)

    lm("Forward fill weekends and holidays")
    logging.debug(df_concat.isnull().sum())
    df_concat.fillna(method='ffill', inplace=True)
    logging.debug(df_concat.isnull().sum())
    return df_concat


def set_date_range(df, start_date, end_date):
    lm("Set date range Dates")
    logging.debug(start_date, end_date)
    # using a 15 year perriod
    lm("Using aa 15 year period of data")
    df = df[start_date:end_date].copy()
    logging.debug(df.head())
    return df

# df_raw = load_and_prepare_data()

# start_date = '2001-01-04'
# end_date = '2016-01-04'
# df_train = set_date_range(df_raw, start_date, end_date)


def calc_daily_ret(df):
    lm("calculate daily returns")
    # calculate dailt returns
    df_dr = (df / df.shift(1)) - 1
    df_dr.columns = ["%s_%s" % (col, 'dr') for col in df.columns]
    df_dr.fillna(method='bfill', inplace=True)
    return df_dr


def calc_rolling_averages(df_in, windows):
    lm("caluclate rolling averages")
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

        logging.debug(cols)
        df_in[cols].plot(figsize=(15, 5)).legend(
            loc='center left', bbox_to_anchor=(1, 0.5))
        plt.axhline(0)
        plt.show()

    lm("Draw a graph")
    draw_graphs(df_dr_rm, df_dr.columns, [180])
    #draw_graphs(df_dr_rm, df_dr.columns, [30])

    #df_all = pd.concat([df, df_dr, df_dr_rm], axis=1)

# df_dr = calc_daily_ret(df_train)
# windows = [2, 7, 30, 180]
# df_rw = calc_rolling_averages(df_dr, windows)

# df_y = create_y_labels(df_dr)
