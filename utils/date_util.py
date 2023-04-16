# -*- coding = utf-8 -*-

from datetime import datetime

import pandas as pd


def calc_bdate(date_str_1, date_str_2):
    # 将日期字符串转换为 datetime 对象
    date_1 = datetime.strptime(date_str_1, "%Y%m%d")
    date_2 = datetime.strptime(date_str_2, "%Y%m%d")
    return len(pd.bdate_range(date_1, date_2))
