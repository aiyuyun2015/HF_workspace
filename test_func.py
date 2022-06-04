import pandas as pd
import numpy as np
import dask
from dask import compute, delayed
import functools
from common import (load, get_sample_ret, parLapply,
                    DATA_PATH, product_list, product,
                    compute, get_daily_pnl_fast,
                    get_daily_pnl, dire, compute_wpr,
                    adf_kpss, sample_for_stationarity,
                    get_performance, float_ndarray_equal,
                    EPSILON, add_bandwidth_2mask)


def test_fast_pnl(all_dates, threshold=0.001, show=True):
    with dask.config.set(scheduler='processes', num_workers=CORE_NUM):
        f_par = functools.partial(get_daily_pnl_fast, product="ru", period=4096,
                                  tranct_ratio=True, threshold=threshold, tranct=1.1e-4,
                                  noise=0)
        result = compute([delayed(f_par)(date) for date in all_dates])[0]
    result = pd.concat(result)
    df1 = get_performance(result, 1, show)
    return df1


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