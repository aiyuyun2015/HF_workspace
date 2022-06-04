#!/usr/bin/env python
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
import functools
from common import (load, get_sample_ret, parLapply,
                    DATA_PATH, product_list, product,
                    compute, get_daily_pnl_fast,
                    get_daily_pnl, dire, compute_wpr,
                    adf_kpss, sample_for_stationarity,
                    get_performance, float_ndarray_equal,
                    EPSILON, add_bandwidth_2mask)
from test_func import (test_fast_pnl, test_wprret_computation, test_diff_in_shift_data )


def test_data():
    # open file
    #date = '20190611'
    input_file = dire + "/" + date + ".pkl"
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





def test_fast_pnl_one_file():
    # test-0
    df0 = get_daily_pnl_fast(all_dates[0], product="ru", period=4096,
                             tranct_ratio=True, threshold=0.001,
                             tranct=1.1e-4)
    if UNITTEST:
        output_file = DATA_PATH + "fast_data.csv"
        assert df0.equals(pd.read_csv(output_file))
        print(df0)


def main():

    test_data()
    exit()
    test_fast_pnl_one_file()

    # test-1, different thresholds result.
    thresholds = [0.001, 0.01, 0.02]
    if DEBUG:
        thresholds = [0.02]

    for thrd in thresholds:
        df = test_fast_pnl(all_dates, thrd)
        print(df)

    exit()
    # test-4
    with dask.config.set(scheduler='processes', num_workers=CORE_NUM):
        f_par = functools.partial(get_daily_pnl, product="ru", period=4096, tranct_ratio=True,
                                  threshold=0.001, tranct=1.1e-4, noise=0)
        result_4 = compute([delayed(f_par)(date) for date in all_dates])[0]

    df4 = pd.DataFrame(get_performance(result_4, 1), index=[0])

    # test-5
    with dask.config.set(scheduler='processes', num_workers=CORE_NUM):
        f_par = functools.partial(get_daily_pnl, product="ru", period=4096, tranct_ratio=True,
                                  threshold=0.002, tranct=1.1e-4, noise=5)
        result_5 = compute([delayed(f_par)(date) for date in all_dates])[0]

    df5 = pd.DataFrame(get_performance(result_5, 1), index=[0])

    get_daily_pnl(all_dates[0], product="ru", period=4096, tranct_ratio=True,
                  threshold=0.001, tranct=1.1e-4, notional=True)

    # test-6
    with dask.config.set(scheduler='processes', num_workers=CORE_NUM):
        f_par = functools.partial(get_daily_pnl, product="ru", period=4096, tranct_ratio=True, threshold=0.001,
                                  tranct=1.1e-4, noise=0, notional=True)
        result = compute([delayed(f_par)(date) for date in all_dates])[0]

    df6 = pd.DataFrame(get_performance(result, 1), index=[0])


if __name__=='__main__':
    UNITTEST = True
    DEBUG = True
    warnings.filterwarnings('ignore')
    os.chdir(DATA_PATH)
    CORE_NUM = int(os.environ['NUMBER_OF_PROCESSORS'])
    os.getcwd()
    all_dates = list(map(lambda x: x, os.listdir(DATA_PATH + product)))
    len(all_dates)
    date = "20190611"
    main()


