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
from test_func import (compute_pnl_with_dask, test_data,
                       test_fast_pnl_one_file)


def test_fast_pnl_all_dates_different_thresholds(thresholds):
    for thrd in thresholds:
        df = compute_pnl_with_dask(all_dates, get_daily_pnl_fast, thrd)
        print(df)

def main():

    # First peek at the dataframe
    test_data(date)

    # Use one day, or half of the file for "fast" pnl test, there is no dask wrapper, or parallelization for it
    # purely calling get_daily_pnl_fast
    test_fast_pnl_one_file(all_dates[0])

    # With different thresholds result. Calling get_daily_pnl_fast, but with dask
    thresholds = [0.001, 0.01, 0.02]
    test_fast_pnl_all_dates_different_thresholds(thresholds)

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
    os.getcwd()
    all_dates = list(map(lambda x: x, os.listdir(DATA_PATH + product)))
    len(all_dates)
    date = "20190611"
    main()


