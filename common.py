import math, os
import numpy as np
import pandas as pd
import dask
import itertools
import matplotlib.pyplot as plt
import functools
from dask import compute, delayed
import _pickle as cPickle
import gzip
import statsmodels.tsa.stattools as ts
import warnings
from MetaData.product_info import *
from collections import OrderedDict

#DATA_PATH = 'C:\\Users\\Administrator\\OneDrive\\Vincent\\enerygy pkl tick 20220303\\'
#DATA_PATH = 'F:\\BaiduNetdiskDownload\\babyquant'
#DATA_PATH = 'C:\\Users\\Administrator\\Documents\\486_disk\\'
DATA_PATH = 'F:\\data_input\\'
SAVE_PATH = 'F:\\data_head\\'
product_list = ["bu", "ru", "v", "pp", "l", "jd"]
product = product_list[0]
dire = os.path.join(DATA_PATH, product)
EPSILON = 1e-5
CORE_NUM = int(os.environ['NUMBER_OF_PROCESSORS'])


def cum(x, n):
    sum_x = x.cumsum()
    sum_x_shift = sum_x.shift(n)
    sum_x_shift[:n]= 0
    return sum_x - sum_x_shift


def drawdown(x):
    y = np.cumsum(x)
    return np.max(y)-np.max(y[-1:])


def max_drawdown(x):
    y = np.cumsum(x)
    return np.max(np.maximum.accumulate(y)-y)


def sharpe(x):
    return zero_divide(np.mean(x)* np.sqrt(250), np.std(x, ddof=1))


def get_hft_summary(result, thre_mat, n):
    all_result = pd.DataFrame(data={"daily.result": result})
    daily_num = all_result['daily.result'].apply(lambda x: x["num"])
    daily_pnl = all_result['daily.result'].apply(lambda x: x["pnl"])
    daily_ret = all_result['daily.result'].apply(lambda x: x["ret"])
    total_num = daily_num.sum()
    if len(total_num) != len(thre_mat):
        raise ValueError("Mismatch!")
    total_pnl = daily_pnl.sum()
    total_ret = daily_ret.sum()
    avg_pnl = zero_divide(total_pnl, total_num)
    avg_ret = zero_divide(total_ret, total_num)
    total_sharp = sharpe(daily_pnl)
    total_drawdown = drawdown(daily_pnl)
    total_max_drawdown = max_drawdown(daily_pnl)
    sharpe_ret = sharpe(daily_ret)
    drawdown_ret = drawdown(daily_ret)
    max_drawdown_ret = max_drawdown(daily_ret)
    final_result = pd.DataFrame(data=OrderedDict([("open", thre_mat["open"]), ("close", thre_mat["close"]), ("num", total_num),
                                                 ("avg.pnl", avg_pnl), ("total.pnl", total_pnl), ("sharpe", total_sharp),
                                                 ("drawdown", total_drawdown), ("max.drawdown", total_max_drawdown),
                                                  ("avg.ret", avg_ret), ("total.ret",total_ret), ("sharpe.ret", sharpe_ret),
                                                  ("drawdown.ret", drawdown_ret), ("max.drawdown.ret", max_drawdown_ret),
                                                 ("mar", total_pnl/total_max_drawdown), ("mar.ret", total_ret/max_drawdown_ret)]),
                                index=thre_mat.index)
    return OrderedDict([("final.result", final_result), ("daily.num", daily_num), ("daily.pnl", daily_pnl), ("daily.ret", daily_ret)])


def get_signal_pnl(file, product, signal_name, thre_mat, reverse=None, tranct=None, max_spread=None, tranct_ratio=None,
                   DATA_PATH=None, SIGNAL_PATH=None, atr_filter=None):
    ## load data
    raw_path = os.path.join(DATA_PATH, product, file)
    data = load(raw_path)
    signal_path = os.path.join(SIGNAL_PATH, 'tmp_pkl', product, signal_name, file)
    S = load(signal_path)
    pred = S * reverse
    pred = pred[data["good"]]
    atr_signal_path = os.path.join(SIGNAL_PATH, 'tmp_pkl', 'atr.4096', signal_name, file)
    atr = load(atr_signal_path)
    atr = atr[data["good"]].reset_index(drop=True)
    data = data[data["good"]].reset_index(drop=True)
    # n_bar = len(data)

    ## load signal

    ## we don't know the signal is positive correlated or negative correlated
    # n_thre = len(thre_mat)
    result = pd.DataFrame(data=OrderedDict([("open", thre_mat["open"].values), ("close", thre_mat["close"].values),
                                            ("num", 0), ("avg.pnl", 0), ("pnl", 0), ("avg.ret", 0), ("ret", 0)]),
                          index=thre_mat.index)
    count = 0
    cur_spread = data["ask"] - data["bid"]
    for thre in thre_mat.iterrows():
        count = count + 1
        buy = pred > thre[1]["open"]
        sell = pred < -thre[1]["open"]
        signal = pd.Series(data=0, index=data.index)
        position = signal.copy()
        signal[buy] = 1
        signal[sell] = -1
        signal[atr < atr_filter] = 0
        scratch = -thre[1]["close"]
        position_pos = pd.Series(data=np.nan, index=data.index)
        position_pos.iloc[0] = 0
        position_pos[(signal == 1) & (data["next.ask"] > 0) & (data["next.bid"] > 0) & (cur_spread < max_spread)] = 1
        position_pos[(pred < -scratch) & (data["next.bid"] > 0) & (cur_spread < max_spread)] = 0
        position_pos.ffill(inplace=True)
        pre_pos = position_pos.shift(1)
        notional_position_pos = pd.Series(data=0, index=data.index)
        notional_position_pos[position_pos == 1] = 1
        notional_position_pos[(position_pos == 1) & (pre_pos == 1)] = np.nan
        notional_position_pos[(notional_position_pos == 1)] = 1 / data["next.ask"][(notional_position_pos == 1)]
        notional_position_pos.ffill(inplace=True)
        position_neg = pd.Series(data=np.nan, index=data.index)
        position_neg.iloc[0] = 0
        position_neg[(signal == -1) & (data["next.ask"] > 0) & (data["next.bid"] > 0) & (cur_spread < max_spread)] = -1
        position_neg[(pred > scratch) & (data["next.ask"] > 0) & (cur_spread < max_spread)] = 0
        position_neg.ffill(inplace=True)
        pre_neg = position_neg.shift(1)
        notional_position_neg = pd.Series(data=0, index=data.index)
        notional_position_neg[position_neg == -1] = -1
        notional_position_neg[(position_neg == -1) & (pre_neg == -1)] = np.nan
        notional_position_neg[(notional_position_neg == -1)] = -1 / data["next.bid"][(notional_position_neg == -1)]
        notional_position_neg.ffill(inplace=True)
        position = position_pos + position_neg
        notional_position = notional_position_pos + notional_position_neg
        # position[n_bar-1] = 0
        position.iloc[0] = 0
        position.iloc[-2:] = 0
        notional_position.iloc[0] = 0
        notional_position.iloc[-2:] = 0
        change_pos = position - position.shift(1)
        notional_change_pos = notional_position - notional_position.shift(1)
        change_pos.iloc[0] = 0
        notional_change_pos.iloc[0] = 0
        change_base = pd.Series(data=0, index=data.index)
        change_buy = change_pos > 0
        change_sell = change_pos < 0
        if (tranct_ratio):
            change_base[change_buy] = data["next.ask"][change_buy] * (1 + tranct)
            change_base[change_sell] = data["next.bid"][change_sell] * (1 - tranct)
        else:
            change_base[change_buy] = data["next.ask"][change_buy] + tranct
            change_base[change_sell] = data["next.bid"][change_sell] - tranct
        final_pnl = -sum(change_base * change_pos)
        ret = -sum(change_base * notional_change_pos)
        num = sum((position != 0) & (change_pos != 0))
        if num == 0:
            result.loc[thre[0], ("num", "avg.pnl", "pnl", "avg.ret", "ret")] = (0, 0, 0, 0, 0)
            return result
        else:
            avg_pnl = np.divide(final_pnl, num)
            avg_ret = np.divide(ret, num)
            result.loc[thre[0], ("num", "avg.pnl", "pnl", "avg.ret", "ret")] = (num, avg_pnl, final_pnl, avg_ret, ret)
    return result


def compute_signal_pnl(product, thre_mat, n_days, all_trade_stat, signal_name, all_dates):
    spread = product_info[product]["spread"]
    tranct = product_info[product]["tranct"]
    tranct_ratio = product_info[product]["tranct.ratio"]
    with dask.config.set(scheduler='processes', num_workers=CORE_NUM):
        f_par = functools.partial(get_signal_pnl, product=product, signal_name=signal_name, thre_mat=thre_mat,
                                  reverse=1, tranct=tranct, max_spread=spread * 1.1, tranct_ratio=tranct_ratio,
                                  atr_filter=0.01, DATA_PATH=DATA_PATH, SIGNAL_PATH=SAVE_PATH)
        result = compute([delayed(f_par)(file) for file in all_dates])[0]
    trade_stat = get_hft_summary(result, thre_mat, n_days)
    all_trade_stat[product] = trade_stat


def make_grid(open_list, close_list):
    product_tuple = list(itertools.product(open_list, close_list))  # make list to use the generator twice
    open_price = list(map(lambda x: x[0], product_tuple))
    close_price = list(map(lambda x: x[0] * x[-1] * (-1), product_tuple))
    thre_mat = pd.DataFrame({'open': open_price, 'close': close_price})
    return thre_mat


def ewma(x, halflife, init=0, adjust=False):
    init_s = pd.Series(data=init)
    s = init_s.append(x)
    if adjust:
        xx = range(len(x))
        lamb = 1 - 0.5**(1 / halflife)
        aa = 1-np.power(1-lamb, xx)*(1-lamb)
        bb = s.ewm(halflife=halflife, adjust=False).mean().iloc[1:]
        return bb/aa
    else:
        return s.ewm(halflife=halflife, adjust=False).mean().iloc[1:]


def vanish_thre(x, thre):
    x[np.abs(x) > thre] = 0
    return x


def zero_divide(x, y):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = np.divide(x, y)
    if hasattr(y, "__len__"):
        res[y == 0] = 0
    elif y == 0:
        if hasattr(x, "__len__"):
            res = np.zeros(len(x))
        else:
            res = 0
    return res


def get_leaves(parent, files, dirs=None, force=True):
    '''
    O(n) version
    Extract all the leaves(files) given a path with recurtion
    :param parent: should be the path, note, use the deepest
    :param files: empty []
    :param dirs: empty []
    :param force: when hit a soft link or sth not file, skip it or stop.
    :return: files
    '''
    basename = os.listdir(parent)
    for i in basename:
        candidate = os.path.join(parent, i)
        if os.path.isfile(candidate):
            files.append(candidate)
        elif os.path.isdir(candidate):
            dirs.append(candidate)
            get_leaves(candidate, files, dirs)
        else:
            if force:
                print(f"Warning: {candidate} not file or path, skip..")
            else:
                raise ValueError(f"{candidate} not file. Stop.")
    return files


def create_signal_path(signal_list, product, HEAD_PATH):
    '''
    Given a path and signal list, create folder has the below tree shape
    HEADP_PATH:
        -temp_pkl
            - product1, e.g., bu
            - jd
            - l
            - pp
            - ru
            - v
                - signal1, e.g., trade.imb.4096
                - signal2
                ...
            ...

    :param signal_list:
    :param product:
    :param HEAD_PATH:
    :return:
    '''

    signal_name = signal_list.factor_name
    periods = signal_list.params
    for key, vals in periods.items():
        for val in vals:
            dirname = HEAD_PATH + "/tmp_pkl/" + product + "/"
            signal_name_specification = signal_name.replace(key, str(val))
            signal_dir = dirname + signal_name_specification
            os.makedirs(signal_dir, exist_ok=True)


def plot_data(data, show=False):
    import matplotlib.pyplot as plt
    plt.figure(1, figsize=(16, 10))
    plt.plot(data.reset_index()["wpr"])
    plt.plot(data.reset_index()[data.reset_index()["good"]]["wpr"])
    if show: plt.show()


def add_bandwidth_2mask(mask, size=5):
    ans = np.zeros_like(mask)
    for i in range(-size, size+1, 1):
        ans = np.logical_or(ans, mask.shift(i, fill_value=False).values)
    return ans


def float_equal(x, y):
    if np.abs(x - y) > EPSILON:
        print(x, y)
    assert np.abs(x - y) < EPSILON


def float_ndarray_equal(*args):
    if len(args) < 1:
        print("Warning: length must greater than 1")
        return True
    args = list(map(lambda x: x.values if isinstance(x, pd.Series) else x, args))

    # Quick check
    diff = (np.abs(args[0] - args[1]) < EPSILON).all()
    if not diff:
        return diff

    # Complete check
    x0 = args[0]
    diffs = [(np.abs(x0 - x) < EPSILON).all() for x in args[1:]]
    return diff and all(diffs)


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


def save(data, path):
    serialized = cPickle.dumps(data)
    with gzip.open(path, 'wb', compresslevel=1) as file_object:
        file_object.write(serialized)


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


class PnlCalculator(object):
    def __init__(self):
        self.tranct_ratio = None
        self.threshold = None
        self.tranct = None
        self.noise = None
        self.ori_data = None
        self.middle_day_points = None
        self.data = {}
        self.n_bar = {}
        self.unit = {}
        self.noise_ret = None

    @staticmethod
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

    def read_from_date(self, date):
        if date in self.data:
            print(f'cached..{date}')
            return self.data[date]
        input_file = dire+"/"+date
        #print(f"open file {input_file}")
        with gzip.open(input_file, 'rb', compresslevel=1) as file_object:
            raw_data = file_object.read()
        temp = cPickle.loads(raw_data)
        mask = temp["good"]
        self.data[date] = temp[mask].reset_index(drop=True)
        self.n_bar[date] = len(self.data[date])
        self.unit[date] = np.std(self.data[date]["ret"])
        return self.data[date]

    def compute_noise(self, date, noise):
        np.random.seed(10)
        self.noise_ret = np.random.normal(scale=self.unit[date] * noise, size=self.n_bar[date])
        return self.noise_ret

    @staticmethod
    def agressive_strategy( data, ret_long, threshold):
        mask_pos = (ret_long > threshold) & (np.array(data["next.ask"]) > 0)
        mask_neg = (ret_long < -threshold) & (np.array(data["next.bid"]) > 0)
        signal = np.where(mask_pos, 1, np.where(mask_neg, -1, 0))
        # convert to series.. for his weird syntax consistency
        position = pd.Series(signal)
        return position

    @staticmethod
    def conservative_strategy(data, ret_long, threshold, capital=None):
        mask_pos = (ret_long > threshold) & (data['next.ask'] > 0) & (data['next.bid'] > 0)
        mask_neg = (ret_long < -threshold) & (data['next.ask'] > 0) & (data['next.bid'] > 0)
        signal = np.where(mask_pos, 1, np.where(mask_neg, -1, np.nan))
        position = pd.Series(signal)
        position = position.fillna(method='ffill').fillna(0)

        if capital:
            # make unchanged position np.nan
            position = np.where(position.diff(1) == 0, np.nan, position)
            position = np.where(position == 1, capital / data['next.ask'],
                                np.where(position == -1,  -capital / data['next.bid'], np.nan))
            position = pd.Series(position)
            position = position.fillna(method='ffill').fillna(0)
        else:
            pass

        return position

    # TODO: think about how to make the arguments aligned, ... we have three PNL computation functions
    def get_daily_pnl_fast(self, date, product="rb", period=4096, tranct_ratio=False, threshold=0.001, tranct=0.21,
                           noise=0, notional=False, capital=None):
        data = self.read_from_date(date)
        # signal has three values: 1, 0, -1 --> price is too low, medium, high
        ret_long = self.compute_ret_and_padding(data, period)
        position = self.agressive_strategy(data, ret_long, threshold)
        result = get_pnl_from_data_positions(data, position, tranct_ratio, tranct,date)
        return result

    def get_daily_pnl(self, date, product=None, period=None, tranct_ratio=None,
                      threshold=None, tranct=None, noise=None, notional=None, capital=None):
        data = self.read_from_date(date)
        noise_ret = self.compute_noise(date, noise)
        ret_long = self.compute_ret_and_padding(data, period) + noise_ret
        position = self.conservative_strategy(data, ret_long, threshold)
        result = get_pnl_from_data_positions(data, position, tranct_ratio, tranct, date)

        return result

    def get_daily_pnl_fixed_capital(self, date, product=None, period=None, tranct_ratio=None,
                                   threshold=None, tranct=None, noise=None, notional=None, capital=None):
        data = self.read_from_date(date)
        noise_ret = self.compute_noise(date, noise)
        ret_long = self.compute_ret_and_padding(data, period) + noise_ret
        position = self.conservative_strategy(data, ret_long, threshold, capital=capital)
        result = get_pnl_from_data_positions(data, position, tranct_ratio, tranct, date)

        return result


def to32int(df, cols):
    for col in cols:
        df[col] = df[col].astype(np.int32)
    return df


def pnl_drawdown(s):
    if (s < 0).all():
        print("Warning: all pnl < 0, meaningless..")
        return None
    # np.min(s/s.cummax() - 1)*100
    return max(s.cummax()-s)/s.iloc[-1]


def get_performance(result, spread=1, show=False):
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
    if show:
        plt.show()
    n_days = len(stat)
    num = stat["num"].mean()
    if num == 0:
        return
    if stat["final.pnl"].std() == 0:
        sharpe = 0
    else:
        sharpe = stat["final.pnl"].mean() / stat["final.pnl"].std() * math.sqrt(250)

    drawdown = pnl_drawdown(stat['cum_pnl'])
    mar = 1 / (drawdown ) if drawdown is not None else None
    win_ratio = sum(stat["final.pnl"] > 0) / n_days

    avg_pnl = sum(stat["final.pnl"]) / sum(stat["num"]) / spread
    hld_period = sum(stat["hld.period"]) / sum(stat["num"])
    summary = {"sharpe": sharpe, "drawdown": drawdown,
                 "mar": mar, "win.ratio": win_ratio, "num": num,
                 "avg.pnl": avg_pnl, "hld.period": hld_period}
    return pd.DataFrame(summary, index=[0]) # pd.DataFrame([summary])


def get_pnl_from_data_positions(data, position, tranct_ratio, tranct, date):
    n_bar = len(data)
    position[0] = 0
    position[n_bar - 1] = 0  # close position before the end of day
    position[n_bar - 2] = 0
    change_position = position.diff(1).fillna(0)  # compare today with prev day, that's the change
    change_base = np.zeros(n_bar)
    change_buy = np.array(change_position > 0)
    change_sell = np.array(change_position < 0)

    if (tranct_ratio):
        change_base[change_buy] = data["next.ask"][change_buy] * (
                    1 + tranct)  ## buy price, use next ask, tranct cost use notional*ratio
        change_base[change_sell] = data["next.bid"][change_sell] * (1 - tranct)  ## sell price use next bid
    else:
        change_base[change_buy] = data["next.ask"][change_buy] + tranct  ## fix tranct cost per share
        change_base[change_sell] = data["next.bid"][change_sell] - tranct

    final_pnl = -sum(change_base * change_position)
    turnover = sum(change_base * abs(change_position))
    num = sum((position != 0) & (change_position != 0))  ## number of trades
    hld_period = sum(position != 0)  ## holding period
    result = pd.DataFrame({"date": date, "final.pnl": final_pnl, "turnover": turnover, "num": num,
                           "hld.period": hld_period}, index=[0])
    return result


from collections import OrderedDict


