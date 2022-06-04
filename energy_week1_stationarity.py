#!/usr/bin/env python
import warnings
import os
import math
import _pickle as cPickle
import gzip
import statsmodels.tsa.stattools as ts
from itertools import chain
from common import (get_sample_ret, parLapply,
                    DATA_PATH, product,
                    dire, compute_wpr, adf_kpss,
                    sample_for_stationarity)


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

    # plain adf, kpss
    print("Run plain adf/kpss")
    adf, kpss = adf_kpss(data)
    print(adf, kpss)
    # sample data, then adf, and kpss
    print("Sample very 120 data points, run adf/kpss")
    sampled_data = sample_for_stationarity(data)
    adf = ts.adfuller(sampled_data, maxlag=int(pow(len(sampled_data)-1, 1/3)), regression='ct', autolag=None)
    kpss = ts.kpss(sampled_data, regression='c', nlags=int(3*math.sqrt(len(sampled_data))/13))
    print(adf, kpss)

    # concat all the files, and also test adf and kpss
    ret_2000 = (data["ret"].rolling(2000).sum()).dropna().reset_index(drop=True)

    # Start to run
    print("Concat all the files, run..")
    result = parLapply(CORE_NUM, all_dates, get_sample_ret, period=4096)
    ret_long = list(chain.from_iterable(result))
    adf_long = ts.adfuller(ret_long, maxlag=int(pow(len(ret_long)-1, 1/3)),
                           regression='ct', autolag=None)
    kpss_long = ts.kpss(ret_long, regression='c',
                        nlags=int(3*math.sqrt(len(ret_long))/13))
    print(adf_long)
    print("="*20)
    print(kpss_long)