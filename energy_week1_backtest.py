#!/usr/bin/env python
import warnings
import os
import numpy as np
from common import (DATA_PATH, product,
                    float_equal,
                    EPSILON, add_bandwidth_2mask, PnlCalculator)
from test_func import (compute_pnl_with_dask, test_data,
                       test_fast_pnl_one_file, test_fixed_size_pnl_one_file,
                       test_fixed_size_pnl_all_files,)


def main():

    # First peek at the dataframe
    if SLOW:
        test_data(all_dates[0])

    # test - 1
    # Use one day, or half of the file for "fast" pnl test, there is no dask wrapper, or parallelization for it
    # simply calling get_daily_pnl_fast
    calc = PnlCalculator()
    test_fast_pnl_one_file(all_dates[0])

    # test -2
    # With different thresholds result. Calling get_daily_pnl_fast, but with dask
    # we can see it's really bad
    # so try increasing threshold..
    if SLOW:
        print("Test fast pnl 0.001")
        compute_pnl_with_dask(all_dates, calc.get_daily_pnl_fast, 0.001, show=SHOW_PLOT)

    # test - 3
    # increase threshold, better
    # Although it's profitable there are very few trades.
    # Now we use a different scheme.
    if SLOW:
        print("Test fast pnl 0.01")
        df = compute_pnl_with_dask(all_dates, calc.get_daily_pnl_fast, 0.01, show=SHOW_PLOT)
        float_equal(df['sharpe'].values[0], -0.070633)
        print("Test fast pnl 0.02")
        df = compute_pnl_with_dask(all_dates, calc.get_daily_pnl_fast, 0.02, show=SHOW_PLOT)
        float_equal(df['sharpe'].values[0], 0.890166)

    # test - 4
    # In previous scheme, we close our position when the value is not strong enough.
    # It may close the positions too soon that it cannot cover transaction cost on average
    # So we change our backtest method to make it holding positions longer

    # Better pnl with conservative strategy, less tradings
    # chaning pnl calculator
    print("Test pnl (conservative) one day")
    test_fixed_size_pnl_one_file(all_dates[0])

    if SLOW:
        print("Test pnl (conservative) 0.001")
        df = compute_pnl_with_dask(all_dates, calc.get_daily_pnl, 0.001, show=SHOW_PLOT)
        float_equal(df['sharpe'].values[0], 27.186233)

    # however, if, we add noise, it becomes worse..
    if SLOW:
        print("Test pnl (conservative) 0.001 with noise=5")
        test_fixed_size_pnl_all_files(all_dates, noise=5, show=True)

    # test-5
    # With fixed capital (1USD)
    # keep notional true, however, never used.
    print("Test capital fixed")
    df = calc.get_daily_pnl_fixed_capital(all_dates[0], product="ru", period=4096,
                                          tranct_ratio=True, threshold=0.001,
                                          tranct=1.1e-4, notional=True, noise=0, capital=1)
    float_equal(df['sharpe'].values[0], 0.04904261)

    df = compute_pnl_with_dask(all_dates, calc.get_daily_pnl_fixed_capital, 0.001,
                               product='rb', period=4096, tranct_ratio=True,
                               tranct=1.1e-4, notional=True,
                               noise=0, capital=1, show=SHOW_PLOT)
    float_equal(df['sharpe'].values[0], 26.503323)


if __name__=='__main__':
    UNITTEST = True
    DEBUG = False
    SLOW = True
    SHOW_PLOT = False
    warnings.filterwarnings('ignore')
    os.chdir(DATA_PATH)
    os.getcwd()
    all_dates = list(map(lambda x: x, os.listdir(DATA_PATH + product)))

    if DEBUG:
        all_dates = all_dates[0:200]

    len(all_dates)
    main()


