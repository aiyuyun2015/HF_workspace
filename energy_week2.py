import os
# from MyLib.helper import *
# from MyLib.stats import *
# from MyLib.product_info import *
# from imp import reload
# import MyLib.helper
# import MyLib.stats
# #import product_info
# reload(MyLib.helper)
# reload(MyLib.stats)
# #reload(product_info)
from collections import OrderedDict
import numpy as np
from common import (parLapply, DATA_PATH, SAVE_PATH, create_signal_path, get_leaves)
from MyLib.SignalClass import (foctor_total_trade_imb_periodv2, build_single_signal)


def main():

    # Create signal path
    x1 = foctor_total_trade_imb_periodv2()
    # for product in product_list:
    #     create_signal_path(x1, product, SAVE_PATH)

    # Generate signal data
    for product in product_list:
        file_list = []
        file_list = get_leaves(DATA_PATH + product, file_list)
        parLapply(CORE_NUM, file_list, build_single_signal,
                  signal_list=x1, product=product, HEAD_PATH=SAVE_PATH)


if __name__=='__main__':

    CORE_NUM = int(os.environ['NUMBER_OF_PROCESSORS'])
    product_list = ["bu", "ru", "v", "pp", "l", "jd"]
    all_dates = sorted(os.listdir(DATA_PATH + product_list[0]))
    # Look-back period
    period = 4096

    main()
