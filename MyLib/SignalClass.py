import os.path
import inspect
import pathlib
import numpy as np
from collections import OrderedDict
from MyLib.helper import (zero_divide, vanish_thre, ewma,)
from common import load, save

VERBOSE=False


class factor_template(object):
    factor_name = ""

    params = OrderedDict([
        ("period", np.power(2, range(10 ,13)))
    ])

    def formula(self):
        pass

    def form_info(self):
        return inspect.getsource(self.formula)

    def info(self):
        info = ""
        info = info + "factor_name:\n"
        info = info +self.factor_name + "\n"
        info = info + "\n"
        info = info + "formula:\n"
        info = info + self.form_info() + "\n"
        info = info + "\n"
        info = info + "params:\n"
        for key in self.params.keys():
            info = info + "$" + key + ":" + str(self.params.get(key)) + "\n"
        return info

    def __repr__(self):
        return self.info()

    def __str__(self):
        return self.info()


class foctor_total_trade_imb_periodv2(factor_template):
    factor_name = "total.trade.imb.period"
    params = {"period": np.power(2, range(12, 13))}

    def formula(self, data, period):
        # buy.trade: active buy volume at level 1
        # buy2.trade: active buy volume at other levels
        # sell.trade: active sell volume at level 1
        # sell2.trade: active sell volume at other levels
        # qty: newest trading volume
        # why level 2 trade could happen? The best guess is, some trades happened, and the price moved up/down or
        # move donw/up within the snapshot data (0.5s)
        # the time diff is about 0.5s~, it's hard to check if the sell.trade (or sell volume) is correct or not
        # overall, if we look at some data, I can see, kind of this is true.. Let's carry on and assume the data is
        # correct.
        try:
            assert (data['buy.trade'] + data['buy2.trade'] + data['sell.trade'] + data['sell2.trade'] == data['qty']).all()
        except Exception as e:
            if VERBOSE:
                print("Warning: volume computation not that correct..")
                mask = data['buy.trade'] + data['buy2.trade'] + data['sell.trade'] + data['sell2.trade'] != data['qty']
                diff = data.loc[mask]
                print(diff[['date.time', 'buy.trade', 'buy2.trade', 'sell.trade', 'sell2.trade']])
            print(e)

        imb = data["buy.trade"] + data["buy2.trade"] - data["sell.trade"] - data["sell2.trade"]
        signal = zero_divide(ewma(imb, period, adjust=True),
                             ewma(data["qty"], period, adjust=True)
                             )

        return vanish_thre(signal, 1).values


def build_single_signal(file_name, signal_list, product, HEAD_PATH, skip=True):
    raw_data = load(file_name)
    basename = os.path.basename(file_name)
    signal_name = signal_list.factor_name
    periods = signal_list.params
    for key, vals in periods.items():
        for val in vals:
            signal_name = signal_name.replace(key, str(val))
            output_dir = os.path.join(HEAD_PATH, "tmp_pkl",  product, signal_name)
            if not os.path.exists(output_dir):
                print(f"Create path {output_dir}")
                pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
            output_file = os.path.join(output_dir, basename)
            if os.path.exists(output_file) and skip:
                print(f"Skip file {output_file}")
                continue
            signal_series = signal_list.formula(raw_data, val)
            print(f"Create file {output_file}")
            save(signal_series, output_file)
