import os
import numpy as np
from common import (parLapply, DATA_PATH, SAVE_PATH, get_leaves, make_grid, compute_signal_pnl)
from MyLib.SignalClass import (FactorTotalTradeImbPeriod, FactorTradeImbPeriod,
                               FactorAtrPeriod, build_single_signal)


def main():

    # Generate signal data
    if not DEBUG:
        for product in PRODUCT_LIST:
            file_list = []
            file_list = get_leaves(DATA_PATH + product, file_list)
            if DEBUG:
                file_list = file_list[0:10]
            parLapply(CORE_NUM, file_list, build_single_signal,
                      signal_list=FactorTotalTradeImbPeriod(),
                      product=product, HEAD_PATH=SAVE_PATH)

            parLapply(CORE_NUM, file_list, build_single_signal,
                      signal_list=FactorTradeImbPeriod(),
                      product=product, HEAD_PATH=SAVE_PATH)

            parLapply(CORE_NUM, file_list, build_single_signal,
                      signal_list=FactorAtrPeriod(),
                      product=product, HEAD_PATH=SAVE_PATH)

    # Make grid for running signal pnl
    open_list = np.arange(0.1, 0.4, 0.02)
    close_list = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
    thre_mat = make_grid(open_list, close_list)

    all_trade_stat = {}
    signal_name = "total.trade.imb.4096"
    for product in PRODUCT_LIST:
        print(product)
        compute_signal_pnl(product, thre_mat, N_DAYS, all_trade_stat, signal_name, ALL_DATES)



if __name__=='__main__':
    DEBUG=False
    CORE_NUM = int(os.environ['NUMBER_OF_PROCESSORS'])
    PRODUCT_LIST = ["bu", "ru", "v", "pp", "l", "jd"]
    ALL_DATES = sorted(os.listdir(DATA_PATH + PRODUCT_LIST[0]))
    N_DAYS = len(ALL_DATES)
    if DEBUG:
        PRODUCT_LIST = PRODUCT_LIST[0:1]
    # Look-back period
    PERIOD = 4096

    main()
