# -*- coding = utf-8 -*-

from datetime import datetime

import pandas as pd
from dateutil.relativedelta import relativedelta


# 计算两个字符串类型的日期之间的工作日数量
def calc_bdate(date_str_1, date_str_2):
    # 将日期字符串转换为 datetime 对象
    date_1 = datetime.strptime(date_str_1, "%Y%m%d")
    date_2 = datetime.strptime(date_str_2, "%Y%m%d")
    return len(pd.bdate_range(date_1, date_2))


# 计算字符串类型日期的前或后指定数量的月份的日期
def gap_period(date_str, months, is_previous):
    if is_previous:
        return datetime.strptime(date_str, "%Y%m%d") - relativedelta(months=months)
    else:
        return datetime.strptime(date_str, "%Y%m%d") + relativedelta(months=months)
