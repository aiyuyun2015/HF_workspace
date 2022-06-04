import math, os
import numpy as np
import pandas as pd
import dask
import matplotlib.pyplot as plt
import functools
from dask import compute, delayed
import _pickle as cPickle
import gzip
import statsmodels.tsa.stattools as ts

DATA_PATH = 'C:\\Users\\Administrator\\OneDrive\\Vincent\\enerygy pkl tick 20220303\\'
#DATA_PATH = 'F:\\BaiduNetdiskDownload\\babyquant'
DATA_PATH = 'C:\\Users\\Administrator\\Documents\\486_disk\\'
product_list = ["bu", "ru", "v", "pp", "l", "jd"]
product = product_list[0]
dire = os.path.join(DATA_PATH, product)


def sample_for_stationarity(data):
    '''
    Sample index, that is 120*n, n=0, 1,2, ..., and also the index belongs to 9:00~23:00
    :param data:
    :return:
    '''

    # np.mod(x1, x2) Returns the element-wise remainder of division.
    sampled_index_every120 = (np.mod(np.arange(0, len(data)), 120) == 0)
    good_index = data['good']
    range_120 = (sampled_index_every120 & good_index)[119:]
    s_idx = np.where(range_120)

    # ret is log(wpr).diff(1) ~ note, there are some approximation, if you want to compare
    # then should you use np.abs(np.log(wpr).diff() - data['ret']) < EPSILON
    # thus, ret_120 = np.log(wpr_120) - np.log(wpr_1)
    # TODO: why reset_index? if you do it, then data, ret_120 are mismatched.
    #   I think this is ok, the bottome line is, the func still looks at good data
    #   plus only look at every 120 data, meaning, no overlap betwen them, maybe this is the better way
    #   to look at the stationarity..
    ret_120 = (data["ret"].rolling(120).sum()).dropna().reset_index(drop=True)
    sampled_data = ret_120.iloc[s_idx]
    return sampled_data


def adf_kpss(data):
    mask_good = data['good']

    # KPSS and ADF test
    data_good = data.loc[mask_good].reset_index(drop=True)
    adf_test = ts.adfuller(data_good["ret"],
                           maxlag=int(pow(len(data_good['ret']) - 1, 1.0 / 3)),
                           regression='ct', autolag=None)

    kpss_test = ts.kpss(data_good['ret'], regression='c',
                        nlags=int(3 * math.sqrt(len(data_good)) / 13))

    return adf_test, kpss_test


def compute_wpr(data):
    wpr = (data["bid"] * data["ask.qty"] + data["ask"] * data["bid.qty"]) / (data["bid.qty"] + data["ask.qty"])
    mask_limit = (data["ask.qty"] == 0) | (data["bid.qty"] == 0)
    wpr = np.where(mask_limit, data["price"], wpr)
    return wpr

def load(path):
    with gzip.open(path, 'rb', compresslevel=1) as file_object:
        raw_data = file_object.read()
    return cPickle.loads(raw_data)


def get_sample_ret(date, period):
    data = load(DATA_PATH + product+"/"+date)
    ret = (data["ret"].rolling(period).sum()).dropna().reset_index(drop=True)
    range = ((np.mod(np.arange(0, len(data)), period) == 0) & data["good"])[(period-1):]
    return ret.iloc[np.where(range)]


def parLapply(CORE_NUM, iterable, func, *args, **kwargs):
    with dask.config.set(scheduler='processes', num_workers=CORE_NUM):
        f_par = functools.partial(func, *args, **kwargs)
        result = compute([delayed(f_par)(item) for item in iterable])[0]
    return result


def compute_ret_and_padding(data, period):
    '''
    Short version, the code computes

    np.log(wpr(t+period) - wpr(t)).fillna(0)

    Long version:
    This seems to be a simple task. But the author is crazy.
    What he did id too complicated, in a weird way. He used ret.rolling.
    Note, ret.rolling = np.log(wpr).diff(). But, he has the data.iloc[0][ret]!! Some missing data is dropped at the
    first place.
    Thus, I cannot simplily refactor the code, with the wrp or wpr.log .. Only dropped some syntax that's not pretty.

    :param data:
    :param period:
    :return:
    '''
    ret = data['ret']
    ret_long = ret.rolling(period).sum().dropna()  ## future return, used as signal
    num_of_zeros = len(data) - len(ret_long)
    zeros = pd.Series([0] * num_of_zeros)
    ret_long = pd.concat([ret_long, zeros])
    return ret_long.reset_index(drop=True)


def get_daily_pnl_fast(date, product="rb", period=4096, tranct_ratio=False, threshold=0.001, tranct=0.21, noise=0):
    '''

    :param date:
    :param product:
    :param period:
    :param tranct_ratio: ratio of transation fee, sometimes it's fixed fee
    :param threshold: how to decide this? at least, long and short position should be similar numbers
    :param tranct: transaction fee? in fast pnl, 0
    :param noise:
    :return:
    '''
    with gzip.open(dire+"/"+date, 'rb', compresslevel=1) as file_object:
        raw_data = file_object.read()
    ori_data = cPickle.loads(raw_data) ## original data
    data = ori_data[ori_data["good"]] ## the middle day of original data
    n_bar = len(data)  ## number of bars
    unit = np.std(data["ret"]) ## standard deviation of return
    np.random.seed(10)

    # signal has three values: 1, 0, -1 --> price is too low, medium, high
    ret_long = compute_ret_and_padding(data, period)
    mask_pos = (ret_long > threshold) & (np.array(data["next.ask"]) > 0)
    mask_neg = (ret_long < -threshold) & (np.array(data["next.bid"]) > 0)
    signal = np.where(mask_pos, 1, np.where(mask_neg, -1, 0))

    # convert to series.. for his weird syntax consistency
    position = pd.Series(signal)

    position[0] = 0
    position[n_bar-1] = 0  # close position before the end of day
    position[n_bar-2] = 0
    change_position = position.diff(1).fillna(0) # compare today with prev day, that's the change
    change_base = np.zeros(n_bar)
    change_buy = np.array(change_position > 0)
    change_sell = np.array(change_position < 0)

    if (tranct_ratio):
        change_base[change_buy] = data["next.ask"][change_buy]*(1+tranct) ## buy price, use next ask, tranct cost use notional*ratio
        change_base[change_sell] = data["next.bid"][change_sell]*(1-tranct) ## sell price use next bid
    else:
        change_base[change_buy] = data["next.ask"][change_buy]+tranct ## fix tranct cost per share
        change_base[change_sell] = data["next.bid"][change_sell]-tranct

    # total pnl, there is a negative sign, because selling get money and buying pay money
    final_pnl = -sum(change_base * change_position)
    turnover = sum(change_base * abs(change_position))
    num = sum((position != 0) & (change_position != 0)) ## number of trades
    hld_period = sum(position != 0)   ## holding period
    result = pd.DataFrame({"date": [date], "final.pnl": [final_pnl], "turnover": [turnover], "num": [num], "hld.period": [hld_period]})
    return result


from collections import OrderedDict


def to32int(df, cols):
    for col in cols:
        df[col] = df[col].astype(np.int32)
    return df


def pnl_drawdown(s):
    if (s < 0).all():
        print("Warning: all pnl < 0, meaningless..")
        return None
    return np.min(s/s.cummax() - 1)*100


def get_performance(result, spread=1):
    # Note1, np.rec.fromrecord, will create a "record array"
    # Note2, result.values --> returns a nd.array, dimension = row * col
    # The usage of record array, is to provide a mimic c-type structure
    # x = np.array([('Rex', 9, 81.0), ('Fido', 3, 27.0)],
    #              dtype=[('name', 'U10'), ('age', 'i4'), ('weight', 'f4')])
    # x
    # array([('Rex', 9, 81.), ('Fido', 3, 27.)],
    #       dtype=[('name', 'U10'), ('age', '<i4'), ('weight', '<f4')])
    # In the above example, as you can see, there are three "objects" saved in the array, first one
    # is name, with less than 10 chars
    # second, age, integer 4*8 = 32 bit
    # thrid float, 4*8 = 32 bit
    stat = result.reset_index()
    stat['date'] = stat['date'].map(lambda x: x.split('.')[0])
    stat['date'] = pd.to_datetime(stat['date'])
    stat = to32int(stat, ['num', 'hld.period'])
    stat['cum_pnl'] = stat["final.pnl"].cumsum()
    plt.figure(1, figsize=(16, 10))
    plt.title("")
    plt.xlabel("date")
    plt.ylabel("pnl")
    plt.plot(stat['date'], stat['cum_pnl'])
    n_days = len(stat)
    num = stat["num"].mean()
    if num == 0:
        return
    if stat["final.pnl"].std() == 0:
        sharpe = 0
    else:
        sharpe = stat["final.pnl"].mean() / stat["final.pnl"].std() * math.sqrt(250)

    drawdown = pnl_drawdown(stat['cum_pnl'])
    mar = 1 / (drawdown / 100) if drawdown is not None else None
    win_ratio = sum(stat["final.pnl"] > 0) / n_days

    avg_pnl = sum(stat["final.pnl"]) / sum(stat["num"]) / spread
    hld_period = sum(stat["hld.period"]) / sum(stat["num"])
    summary = {"sharpe": sharpe, "drawdown(%)": drawdown,
                 "mar": mar, "win.ratio": win_ratio, "num": num,
                 "avg.pnl": avg_pnl, "hld.period": hld_period}
    return pd.DataFrame(summary, index=[0]) # pd.DataFrame([summary])


def get_daily_pnl(date, product="rb", period=2000, tranct_ratio=False, threshold=0.001, tranct=1.1e-4, noise=0):
    with gzip.open(dire + "/" + date, 'rb', compresslevel=1) as file_object:
        raw_data = file_object.read()
    data = cPickle.loads(raw_data)
    data = data[data["good"]].reset_index(drop=True)
    n_bar = len(data)
    unit = np.std(data["ret"])
    np.random.seed(10)
    noise_ret = np.random.normal(scale=unit * noise, size=n_bar)
    ##  we repeat the above code to get daily result
    ret_2000 = (data["ret"].rolling(period).sum()).dropna().reset_index(drop=True)
    ret_2000 = ret_2000.append(pd.Series([0] * (len(data) - len(ret_2000)))).reset_index(drop=True) + noise_ret
    signal = pd.Series([0] * n_bar)
    signal[ret_2000 > threshold] = 1  #
    signal[ret_2000 < -threshold] = -1
    position_pos = pd.Series([np.nan] * n_bar)
    position_pos[0] = 0
    position_pos[(signal == 1) & (data["next.ask"] > 0) & (data["next.bid"] > 0)] = 1  ## if signal==1, position_pos=1
    position_pos[(ret_2000 < -threshold) & (data["next.bid"] > 0)] = 0  ## if ret< -threshold, position_pos=0
    position_pos.ffill(inplace=True)
    position_neg = pd.Series([np.nan] * n_bar)
    position_neg[0] = 0
    position_neg[
        (signal == -1) & (data["next.ask"] > 0) & (data["next.bid"] > 0)] = -1  ## if signal==-1, position_neg=-1
    position_neg[(ret_2000 > threshold) & (data["next.ask"] > 0)] = 0  ## if ret> threshold, position_neg=0
    position_neg.ffill(inplace=True)
    position = position_pos + position_neg  ## total position
    position[0] = 0
    position[n_bar - 1] = 0
    position[n_bar - 2] = 0
    change_pos = position - position.shift(1)
    change_pos[0] = 0
    change_base = pd.Series([0] * n_bar)
    change_buy = change_pos > 0
    change_sell = change_pos < 0
    if (tranct_ratio):
        change_base[change_buy] = data["next.ask"][change_buy] * (1 + tranct)
        change_base[change_sell] = data["next.bid"][change_sell] * (1 - tranct)
    else:
        change_base[change_buy] = data["next.ask"][change_buy] + tranct
        change_base[change_sell] = data["next.bid"][change_sell] - tranct
    final_pnl = -sum(change_base * change_pos)
    turnover = sum(change_base * abs(change_pos))
    num = sum((position != 0) & (change_pos != 0))
    hld_period = sum(position != 0)

    ## finally we combine the statistics into a data frame
    # result = pd.DataFrame({"final.pnl": final_pnl, "turnover": turnover, "num": num, "hld.period": hld_period}, index=[0])
    # result = {"date": date, "final.pnl": final_pnl, "turnover": turnover, "num": num, "hld.period": hld_period}
    result = OrderedDict(
        [("date", date), ("final.pnl", final_pnl), ("turnover", turnover), ("num", num), ("hld.period", hld_period)])
    return result


def get_daily_pnl(date, product="rb", period=2000, tranct_ratio=False, threshold=0.001, tranct=1.1e-4, noise=0,
                  notional=False):
    with gzip.open(dire + "/" + date, 'rb', compresslevel=1) as file_object:
        raw_data = file_object.read()
    data = cPickle.loads(raw_data)
    data = data[data["good"]].reset_index(drop=True)
    n_bar = len(data)
    unit = np.std(data["ret"])
    np.random.seed(10)
    noise_ret = np.random.normal(scale=unit * noise, size=n_bar)
    ##  we repeat the above code to get daily result
    ret_2000 = (data["ret"].rolling(period).sum()).dropna().reset_index(drop=True)
    ret_2000 = ret_2000.append(pd.Series([0] * (len(data) - len(ret_2000)))).reset_index(drop=True) + noise_ret
    signal = pd.Series([0] * n_bar)
    signal[ret_2000 > threshold] = 1
    signal[ret_2000 < -threshold] = -1
    position_pos = pd.Series([np.nan] * n_bar)
    position_pos[0] = 0
    position_pos[(signal == 1) & (data["next.ask"] > 0) & (data["next.bid"] > 0)] = 1
    position_pos[(ret_2000 < -threshold) & (data["next.bid"] > 0)] = 0
    position_pos.ffill(inplace=True)
    pre_pos = position_pos.shift(1)
    position_pos[(position_pos == 1) & (pre_pos == 1)] = np.nan  ## holding positio rather than trade, change to nan
    position_pos[(position_pos == 1)] = 1 / data["next.ask"][(position_pos == 1)]  ## use 1/price as trading volume
    position_pos.ffill(inplace=True)
    position_neg = pd.Series([np.nan] * n_bar)
    position_neg[0] = 0
    position_neg[(signal == -1) & (data["next.ask"] > 0) & (data["next.bid"] > 0)] = -1
    position_neg[(ret_2000 > threshold) & (data["next.ask"] > 0)] = 0
    position_neg.ffill(inplace=True)
    pre_neg = position_neg.shift(1)
    position_neg[(position_neg == -1) & (pre_neg == -1)] = np.nan  ## holding positio rather than trade, change to nan
    position_neg[(position_neg == -1)] = -1 / data["next.bid"][(position_neg == -1)]  ## use 1/price as trading volume
    position_neg.ffill(inplace=True)  ## replace nan by trading volume
    position = position_pos + position_neg
    position[0] = 0
    position[n_bar - 1] = 0
    position[n_bar - 2] = 0
    change_pos = position - position.shift(1)
    change_pos[0] = 0
    change_base = pd.Series([0] * n_bar)
    change_buy = change_pos > 0
    change_sell = change_pos < 0

    if (tranct_ratio):
        change_base[change_buy] = data["next.ask"][change_buy] * (1 + tranct)
        change_base[change_sell] = data["next.bid"][change_sell] * (1 - tranct)
    else:
        change_base[change_buy] = data["next.ask"][change_buy] + tranct
        change_base[change_sell] = data["next.bid"][change_sell] - tranct
    final_pnl = -sum(change_base * change_pos)
    turnover = sum(change_base * abs(change_pos))
    num = sum((position != 0) & (change_pos != 0))
    hld_period = sum(position != 0)

    ## finally we combine the statistics into a data frame
    # result = pd.DataFrame({"final.pnl": final_pnl, "turnover": turnover, "num": num, "hld.period": hld_period}, index=[0])
    # result = {"date": date, "final.pnl": final_pnl, "turnover": turnover, "num": num, "hld.period": hld_period}
    result = OrderedDict(
        [("date", date), ("final.pnl", final_pnl), ("turnover", turnover), ("num", num), ("hld.period", hld_period)])
    return result

