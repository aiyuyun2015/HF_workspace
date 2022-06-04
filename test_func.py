import warnings
import os
import pandas as pd
import numpy as np
import math
import _pickle as cPickle
import gzip
import matplotlib.pyplot as plt
import statsmodels.tsa.stattools as ts
import functools
import dask
from dask import compute, delayed
from itertools import chain
from collections import OrderedDict
from common import (load, get_sample_ret, parLapply,
                    DATA_PATH, product_list, product,
                    compute, dire, compute_wpr,
                    adf_kpss, sample_for_stationarity,
                    get_performance, float_ndarray_equal,
                    EPSILON, add_bandwidth_2mask, PnlCalculator)

CORE_NUM = int(os.environ['NUMBER_OF_PROCESSORS'])


def compute_pnl_with_dask(all_dates, pnl_calculator, threshold,
                  product="rb", period=4096,
                  tranct_ratio=True, tranct=1.1e-4,
                  noise=0, show=True, notional=False):
    with dask.config.set(scheduler='processes', num_workers=CORE_NUM):
        f_par = functools.partial(pnl_calculator, product=product, period=period,
                                  tranct_ratio=tranct_ratio, threshold=threshold, tranct=tranct,
                                  noise=noise, notional=notional)
        result = compute([delayed(f_par)(date) for date in all_dates])[0]
    result = pd.concat(result)
    df = get_performance(result, 1, show)
    print(df)
    return df


def test_wprret_computation(data, wpr_ret, col, verbose=False):
    '''
    The return computation is not that accurate, guess the "raw" data is not raw in some sense, one data
    point is not matched, fine..
    Also, interesting thing is, the data and date.time alignment, the author used a bit different way of
    saving data. Overall it's fine. Just remember to filter by "good" column.
    :param data:
    :return:
    '''

    data['date.time'] = pd.to_datetime(data['date.time'])
    if verbose:
        print("data time range", data['date.time'].min(), data['date.time'].max())
    # NOTE: the author's way of saving data creates duplicated data, suppose, the date is
    # 2019-06-11, then he has
    # 06-10: 21:00 night session, 06-11 day session 9:00, 06-11 night session 21:00, and 06-12 day session 9:00
    # 06-11: again..

    # However, if we only use the data with data['good'] is True, it's fine
    # the file name (date, e.g., 2019-06-11) will include only 2019-06-11 9:00 to 2019-06-23:00
    if verbose:
        print("Data in good:")
    in_use = data['good']
    if verbose:
        print("data time range", data.loc[in_use]['date.time'].min(), data.loc[in_use]['date.time'].max())

    mask = (data['date.time'].diff(1) > pd.to_timedelta(1, unit='min')) | (data.index == 0)
    data_first_trade_in_opening = data.loc[mask][['date.time', col]]
    wpr_first_trade_in_opening = wpr_ret.loc[mask]
    if verbose:
        print(data_first_trade_in_opening)
        print(wpr_first_trade_in_opening)
    if not float_ndarray_equal(data_first_trade_in_opening[col], wpr_first_trade_in_opening):
        if verbose:
            print("Manually checking, eyeballing..? Find the difference.")

    sub_wpr_ret = wpr_ret.loc[~mask]
    sub_data = data[col].loc[~mask]
    assert float_ndarray_equal(sub_wpr_ret, sub_data )


def test_diff_in_shift_data(data, col):

    next_ask = data[col].shift(-1)
    # assert float_ndarray_equal(next_ask, data['next.ask'])
    mask = np.abs(data[f'next.{col}'] - next_ask) > EPSILON
    if False:
        mask = add_bandwidth_2mask(mask)
    df = data.loc[mask][['date.time', f'next.{col}']]
    print(df)
    print(next_ask.loc[mask])


def test_data(date):
    # open file
    #date = '20190611'
    input_file = dire + "/" + date
    with gzip.open(input_file, 'rb', compresslevel=1) as file_object:
        raw_data = file_object.read()
    print(f"Load {input_file}")
    data = cPickle.loads(raw_data)
    wpr = compute_wpr(data)
    log_price = np.log(data['wpr'])
    mid_price = (data['bid'] + data['ask'])/2.0
    assert float_ndarray_equal(wpr, data['wpr'])
    assert float_ndarray_equal(log_price, data['log.price'])
    assert float_ndarray_equal(mid_price, data['mid.price'])

    # Next checking ret with functions.. need to mask the first trade in the opening
    # The conclusion is, there is only one data point 15:00 has issue, weird. Cuz???
    # TODO: maybe think about why 15:00 has issue.
    # wpr.ret is a bit tricky. Note wpr.ret = wpr.diff(1)
    # The annoying part is the first data point in the opening of
    # trade session, overall it's fine after checkings; also, take a look at the way how the data is saved.
    wpr_ret = data['wpr'].diff(1)
    test_wprret_computation(data, wpr_ret, 'wpr.ret', verbose=False)

    # ret defined as np.log(wpr).diff(1)
    log_wpr_ret = np.log(data['wpr']).diff(1)
    test_wprret_computation(data, log_wpr_ret, 'ret', verbose=False)

    # next.ask, next.bid is simply the ask and bid price shift(-1)
    test_diff_in_shift_data(data, 'ask')
    test_diff_in_shift_data(data, 'bid')

    # TODO: check min.1024, etc.
    # min_1024 = data['wpr'].rolling(1024).min()


def test_fast_pnl_one_file(date):
    # test-0
    calc = PnlCalculator()
    df0 = calc.get_daily_pnl_fast(date, product="rb", period=4096,
                             tranct_ratio=True, threshold=0.001,
                             tranct=1.1e-4)

    output_file = DATA_PATH + "rb_fast_data_" + date.split('.')[0]+".csv"
    print(f"Open unit test input:{output_file}")
    assert df0.equals(pd.read_csv(output_file))
    print(df0)


def test_fixed_size_pnl_one_file(date):
    # test-0
    calc = PnlCalculator()
    df0 = calc.get_daily_pnl(date, product="rb", period=4096,
                             tranct_ratio=True, threshold=0.001,
                             tranct=1.1e-4, noise=0, notional=None)

    output_file = DATA_PATH + "rb_fixed_size_pnl_data_" + date.split('.')[0]+".csv"
    expected_df = pd.read_csv(output_file)
    print(f"Open unit test input:{output_file}")
    if not df0.equals(expected_df):
        print("Expected:\n", expected_df)
        print("Calculated:\n", df0)
    assert df0.equals(expected_df)
    print("Test ok.")


def test_fixed_size_pnl_all_files(all_dates, noise=None, show=True):
    calc = PnlCalculator()
    df = compute_pnl_with_dask(all_dates, calc.get_daily_pnl, 0.002, period=4096,
                               tranct_ratio=True, tranct=1.1e-4,
                               noise=noise, show=show)
    output_file = "pnl_test_with_noise.csv"
    if not os.path.exists(output_file):
        df.to_csv(output_file, index=False)
    else:
        expected_df = pd.read_csv(output_file)
        if not df.equals(expected_df):
            print("Expected:\n", expected_df)
            print("Calculated:\n", df)
        assert df.equals(expected_df)