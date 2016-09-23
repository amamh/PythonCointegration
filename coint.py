# import statsmodels.api as sm
# import matplotlib.dates as mdates
# from pandas.tseries.offsets import BDay
# import scipy
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt
import math
import numpy as np
import pandas as pd
import time as t
import quandl as ql
from decorations import timeit, logit
from statsmodels.tsa.stattools import adfuller
from os import path

ql.ApiConfig.api_key = 'zz-ESkkFKLN_u25hEbYX'


# import statsmodels.regression.linear_model as linear_model

def clear_log(logfile='main.log'):
    import os
    if os.path.isfile(logfile):
        os.remove(logfile)


# create filenames and output to csvs
def path_with_time(fn):
    time = dt.datetime.now().strftime("%Y-%m-%d_%H-%M_")
    base_dir = "./"
    return base_dir + time + fn + ".csv"


# Linear regression, will return a new dataframe with the output, error, delta error columns appended.
# This uses interception and rolling window
@timeit()
@logit()
def get_linear_regression_results(x, y, intercept=False, rolling_window=-1):
    """
    :type x: pd.Series
    :type y: pd.Series
    """
    if rolling_window != -1:
        model = pd.ols(x=x, y=y, intercept=intercept, window=rolling_window, window_type='rolling')
    else:
        model = pd.ols(x=x, y=y, intercept=intercept)

    # The type of 'beta' actually changes.. if rolling then DataFrame, else Series
    slope = model.beta['x'] if rolling_window != -1 else model.beta[0]
    const = model.beta['intercept'] if intercept else 0

    result = pd.concat([x, y], axis=1)

    # remove the start that cannot be fit using rolling window mode:
    if rolling_window != -1:
        result = result.iloc[rolling_window - 1:, :]

    result.loc[:, 'slope'] = slope
    result.loc[:, 'const'] = const
    result.loc[:, 'f(x)'] = result.iloc[:, 0] * slope + const
    result.loc[:, 'error'] = result.loc[:, 'f(x)'] - result.iloc[:, 1]
    return result


# @timeit()
# @logit()
# def calc_t_score(series):
#     series_d = series.shift(-1) - series # delta

#     model = pd.ols(x=series, y=series_d)
#     tstat = model.beta[0] / model.std_err[0] # slope / str err (slope)
#     calc_adfuller(series)
#     # return coint score
#     return tstat
# Builtin ADF test is 4 times faster, simpler and gives exact same result, so we use that

# Check if last price in the list (i.e. today's price) has deviated significantly
# This will return None if it's not significant
@timeit()
@logit()
def create_summary(symbol_from, symbol_to, adfresults, y, fx, error=None):
    if error is None:
        error = fx - y

    error_std = error.std()
    # This is effectively x scaled up to the level of y. f(x) = multiplier*x ~= y
    last_scaled_x = fx.iloc[-1]
    last_y = y.iloc[-1]
    last_gap = error.iloc[-1]

    std_no = last_gap / error_std  # how many stds
    spread = last_gap / math.fabs(last_scaled_x)
    std_pct = error_std / last_y
    # error_std[0] means return a number without [] around it (if usually it would be a list)
    output = pd.Series(
        index=['Symbol 1', 'Symbol 2', 'Coint score', '95% coint score', 'Std Dev %', '# Std Dev', 'Spread %'],
        data=[symbol_from, symbol_to, adfresults[0], adfresults[4]['5%'], std_pct, std_no, spread],
        name="{0} from {1}".format(symbol_to, symbol_from)
    )
    return output


@timeit()
@logit()
def extract_graph_data(pair_df, name1, name2):
    data = pair_df.loc[:, [name1, name2, 'f(x)', 'error']]  # x, y, f(x), error
    return data.rename(columns={'f(x)': '{0} from {1}'.format(name2, name1)})


@timeit()
@logit()
# @persist_to_file()
def get_stock_ql(code):
    # TODO: Get date = now - 1 year
    onlinedata = ql.get(code, start_date="2015-06-16")
    s = pd.DataFrame(onlinedata)['Close']
    s.name = code
    return s


def run_coint(stock_names, folder_to_save):
    clear_log('coint.log')

    # dataframe holding date/price for each stock
    price_df = pd.concat([get_stock_ql(stock) for stock in stock_names], axis=1)  # type: pd.DataFrame
    # coint score per pair
    coint_df = pd.DataFrame(index=stock_names, columns=stock_names)
    # ticker1 price, ticker2 price, ticker2 price from ticker1 price, error
    graph_df = pd.DataFrame()
    # This will hold all supposedly profitable pairs for today
    summary_df = pd.DataFrame()

    # paths to plots
    plots_paths = []
    plots_names = []


    for i in range(len(stock_names)):
        for j in range(len(stock_names)):
            # don't repeat the same pair
            if i < j:
                # select two names and create pair
                symbol_from = stock_names[i]
                symbol_to = stock_names[j]
                print('Starting: ' + symbol_from + ' : ' + symbol_to)
                # drop NA together
                pair_price_df = price_df[[symbol_from, symbol_to]].copy()  # type: pd.DataFrame
                pair_price_df.dropna(inplace=True)

                # linear regression
                # TODO: Why should we do rolling regression?
                regression_data_df =\
                    get_linear_regression_results(pair_price_df.iloc[:, 0], pair_price_df.iloc[:, 1]) # , True, 60)

                # run coint test and store value into array
                adf = adfuller(regression_data_df['error'])
                coint_score = adf[0]
                coint_df.loc[symbol_from, symbol_to] = coint_score

                # check if probability is > 95%
                if coint_score < adf[4]['5%']:
                    # manually force the next step for debugging
                    regression_data_df.ix[-1, 'error'] = 5

                    print('\n\nCointegrated pair found: {0}\t\t{1}\n'.format(symbol_from, symbol_to))
                    # check if today there is a significant deviation
                    pair_summary_df = create_summary(symbol_from, symbol_to,
                                                     adf,
                                                     regression_data_df.loc[:, symbol_to],
                                                     regression_data_df.loc[:, 'f(x)'],
                                                     regression_data_df.loc[:, 'error'])
                    error = regression_data_df.loc[:, 'error']
                    # two std away from the mean:
                    if math.fabs(error.iloc[-1]) > (error.mean() + 2 * error.std()):
                        print "Significant deviation found today on {0}\t\t{1}".format(symbol_from, symbol_to)

                        summary_df = summary_df.append(pair_summary_df, ignore_index=True)
                        pair_graph_data = extract_graph_data(regression_data_df, symbol_from, symbol_to)

                        # FIXME: horrible hack
                        pair_graph_data.plot()

                        plot_path = path.join(folder_to_save, "plot_{0}.png".format(i))
                        plots_paths.append(plot_path)
                        plt.savefig(plot_path)

                        plots_names.append("{0} from {1}".format(symbol_to, symbol_from))

                        # graph_df = pd.concat([graph_df, pair_graph_data], axis=1)

                print('Completed: {0} : {1}'.format(symbol_from, symbol_to))

    f = plt.figure()
    sns.heatmap(coint_df.fillna(0), mask=(pd.isnull(coint_df)), cmap="YlGnBu_r")  # have to pass a mask to ignore missing
    plt.yticks(rotation='horizontal')

    f.savefig(path.join(folder_to_save, "heatmap.png"))

    return [summary_df, zip(plots_names, plots_paths)]


    # # format values
    # if not summary_df.empty:
    #     summary_df['Coint score'] = summary_df['Coint score'].map(lambda x: '%.2f' % x)
    #     summary_df['# Std Dev'] = summary_df['# Std Dev'].map(lambda x: '%.2f' % x)
    #     summary_df['Spread %'] = summary_df['Spread %'].map(lambda x: round(float(100) * x, 2))
    #     summary_df['Std Dev %'] = summary_df['Std Dev %'].map(lambda x: round(float(100) * x, 2))

    # print(summary_df)


    # hm = output.ix[:,['Coint Val']].astype(float)#
    # fig,axn = plt.subplots(2, 1, sharex=True, sharey=True)
    # sns.heatmap(hm, annot=True, fmt="f", ax=ax)
    # hm = output.ix[:,['# Std Dev']].astype(float)#
    # sns.heatmap(pd.DataFrame(hm, annot=True, fmt="d" ))
    # sns.heatmap(hm, annot=True, fmt="f", ax=ax)
    # fig,axn = plt.subplots(1, 4, sharex=True, sharey=True)
    # for ax in axn.flat:
    # sns.heatmap(hm, ax=ax)
    # plt.show()




if __name__ == '__main__':
    clear_log()
    start_time = t.time()

    # global variable
    # regression_window = 150
    # vol_days = 150
    # vol_mul = np.sqrt(vol_days)

    #  get in the sector data
    # indexNames = ['SX3P Index', 'SX4P Index', 'SX6P Index', 'SX7P Index', 'SX8P Index', 'SX86P Index', 'SXAP Index',
    #               'SXDP Index', 'SXEP Index', 'SXFP Index', 'SXKP Index', 'SXMP Index', 'SXNP Index', 'SXOP Index',
    #               'SXPP Index', 'SXQP Index', 'SXRP Index', 'SXTP Index', 'SXIP Index', 'SX5E Index', 'DAX Index',
    #               'SXXP Index']
    # stocksNames = ['LSE/AMER', 'LSE/BLVN','LSE/FOG','LSE/NTQ']

    # Good example: EWA and EWC ETFs
    # https://www.quandl.com/data/GOOG/NYSE_EWA-iShares-MSCI-Australia-Index-Fund-EWA
    # https://www.quandl.com/data/GOOG/NYSEARCA_EWC-iShares-MSCI-Canada-Index-ETF-EWC
    stock_names = ["GOOG/NYSE_EWA", "GOOG/NYSEARCA_EWC", "GOOG/NYSEARCA_IGE"]
    # "LSE/FRR","LSE/FRI","LSE/GTC","LSE/GED","LSE/GRPH","LSE/GDL","LSE/GPX","LSE/HNL","LSE/HAIK","LSE/HUR","LSE/HYR","LSE/IGAS","LSE/IKA","LSE/IRG","LSE/IOG","LSE/INFA","LSE/IOF","LSE/IAE","LSE/ITM","LSE/JOG"]





    # coint_df.to_csv(path('coint'))
    # summary_df.to_csv(path('spreads'))
    # graph_df.to_csv(path('graph'))

    print("--- %s seconds ---" % (t.time() - start_time))