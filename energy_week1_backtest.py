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
                    get_performance)


def test_fast_pnl(all_dates, threshold=0.001):
    with dask.config.set(scheduler='processes', num_workers=CORE_NUM):
        f_par = functools.partial(get_daily_pnl_fast, product="ru", period=4096,
                                  tranct_ratio=True, threshold=threshold, tranct=1.1e-4,
                                  noise=0)
        result = compute([delayed(f_par)(date) for date in all_dates])[0]
    result = pd.concat(result)
    df1 = get_performance(result, 1)
    return df1

if __name__=='__main__':
    warnings.filterwarnings('ignore')
    os.chdir(DATA_PATH)
    CORE_NUM = int(os.environ['NUMBER_OF_PROCESSORS'])
    os.getcwd()
    all_dates = list(map(lambda x: x, os.listdir(DATA_PATH + product)))
    len(all_dates)
    date = "20190611"

    # open file
    with gzip.open(dire+"/"+date+".pkl", 'rb', compresslevel=1) as file_object:
        raw_data = file_object.read()
    data = cPickle.loads(raw_data)
    wpr = compute_wpr(data)

    # test-0
    df0 = get_daily_pnl_fast(all_dates[0], product="ru", period=4096,
                                tranct_ratio=True, threshold=0.001,
                                tranct=1.1e-4)
    output_file = DATA_PATH + "fast_data.csv"
    assert df0.equals(pd.read_csv(output_file))
    print(df0)
    #df0.to_csv(output_file, index=False)

    # test-1

    df1 = test_fast_pnl(all_dates, 0.001)
    print(df1)
    assert (df1['sharpe'].values[0] - (-1.573017)) < 0.0001
    exit()
    # test-2
    with dask.config.set(scheduler='processes', num_workers=CORE_NUM):
        f_par = functools.partial(get_daily_pnl_fast, product="ru", period=4096,
                                  tranct_ratio=True, threshold=0.01, tranct=1.1e-4,
                                  noise=0)
        result_2 = compute([delayed(f_par)(date) for date in all_dates])[0]

    df2 = pd.DataFrame(get_performance(result_2), index=[0])

    # test-3
    with dask.config.set(scheduler='processes', num_workers=CORE_NUM):
        f_par = functools.partial(get_daily_pnl_fast, product="ru", period=4096,
                                  tranct_ratio=True, threshold=0.02, tranct=1.1e-4, noise=0)
        result_3 = compute([delayed(f_par)(date) for date in all_dates])[0]

    df3 = pd.DataFrame(get_performance(result_3, 1), index=[0])

    # test-4
    with dask.config.set(scheduler='processes', num_workers=CORE_NUM):
        f_par = functools.partial(get_daily_pnl, product="ru", period=4096, tranct_ratio=True,
                                  threshold=0.001, tranct=1.1e-4, noise=0)
        result_4 = compute([delayed(f_par)(date) for date in all_dates])[0]


    df4 = pd.DataFrame(get_performance(result_4,1), index=[0])

    # test-5
    with dask.config.set(scheduler='processes', num_workers=CORE_NUM):
        f_par = functools.partial(get_daily_pnl, product="ru", period=4096, tranct_ratio=True,
                                  threshold=0.002, tranct=1.1e-4, noise=5)
        result_5 = compute([delayed(f_par)(date) for date in all_dates])[0]

    df5 = pd.DataFrame(get_performance(result_5,1), index=[0])


    get_daily_pnl(all_dates[0], product="ru", period=4096, tranct_ratio=True,
                  threshold=0.001, tranct=1.1e-4, notional=True)

    # test-6
    with dask.config.set(scheduler='processes', num_workers=CORE_NUM):
        f_par = functools.partial(get_daily_pnl, product="ru", period=4096, tranct_ratio=True, threshold=0.001,
                                  tranct=1.1e-4, noise=0, notional=True)
        result = compute([delayed(f_par)(date) for date in all_dates])[0]



    df6 = pd.DataFrame(get_performance(result,1), index=[0])




