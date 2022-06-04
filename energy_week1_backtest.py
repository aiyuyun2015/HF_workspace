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
                    compute, dire, compute_wpr,
                    adf_kpss, sample_for_stationarity,
                    get_performance, float_ndarray_equal,
                    EPSILON, add_bandwidth_2mask, PnlCalculator)
from test_func import (compute_pnl_with_dask, test_data,
                       test_fast_pnl_one_file, test_fixed_size_pnl_one_file)


def main():
    # if DEBUG:
    #     global all_dates
    #     all_dates = all_dates[0:200]
    # First peek at the dataframe
    if False:
        test_data(all_dates[0])

    # Use one day, or half of the file for "fast" pnl test, there is no dask wrapper, or parallelization for it
    # simply calling get_daily_pnl_fast

    test_fast_pnl_one_file(all_dates[0])

    # With different thresholds result. Calling get_daily_pnl_fast, but with dask
    # we can see it's really bad
    # we can try increasing threshold
    #      sharpe   drawdown(%)       mar  win.ratio        num   avg.pnl  hld.period
    # 0 -1.573017 -10089.004849 -0.009912   0.206379  91.104128 -2.008094  270.726907
    if False:
        compute_pnl_with_dask(all_dates, get_daily_pnl_fast, 0.001)

    # increase threshold, better
    #      sharpe   drawdown(%)      mar  win.ratio        num   avg.pnl  hld.period
    # 0 -0.070633 -43661.478574 -0.00229   0.157598  13.758912 -0.302454  154.373219
    if False:
        compute_pnl_with_dask(all_dates, get_daily_pnl_fast, 0.01)

    # Although it's profitable there are very few trades.
    # Now we use a different scheme.
    # In previous scheme, we close our position when the value is not strong enough.
    # It may close the positions too soon that it cannot cover transaction cost on average
    # So we change our backtest method to make it holding positions longer
    #      sharpe  drawdown(%)       mar  win.ratio       num   avg.pnl  hld.period
    # 0  0.890166  -2671.86471 -0.037427   0.047842  2.073171  9.972816  119.866968
    if False:
        compute_pnl_with_dask(all_dates, get_daily_pnl_fast, 0.02)


    test_fixed_size_pnl_one_file(all_dates[0])
    calc = PnlCalculator()
    compute_pnl_with_dask(all_dates, calc.get_daily_pnl, 0.001, product='rb' )


    exit()
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
    os.getcwd()
    all_dates = list(map(lambda x: x, os.listdir(DATA_PATH + product)))
    len(all_dates)
    #date = "20190611"
    main()


