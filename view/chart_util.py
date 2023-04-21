# -*- coding = utf-8 -*-
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st


def get_chart(date_arr, y1, y2, m):
    # 生成日期对应的真实值和预测值数据框
    date_arr = pd.to_datetime(date_arr)
    date_arr2 = date_arr[m:]
    date_offset = pd.offsets.BDay(n=m)
    date_arr_extend = pd.date_range(date_arr2[-1] + date_offset, periods=m, freq='B')
    date_arr2 = np.concatenate((date_arr2, date_arr_extend))
    df1 = pd.DataFrame({'date': date_arr, 'actual_value': y1})
    df2 = pd.DataFrame({'date': date_arr2, 'predict_value': y2})
    data = pd.merge(df1, df2, on='date', how='outer')

    # 创建基础图表
    base = alt.Chart(data).encode(x=alt.X("date:T", axis=alt.Axis(format="%Y/%m/%d")))
    # 创建折线图层
    line_actual = base.mark_line(color="blue").encode(y="actual_value")
    line_predict = base.mark_line(color="orange").encode(y="predict_value")
    # 创建交互层
    selection = alt.selection_single(fields=["date"], nearest=True, on="mouseover", empty="none")
    points_actual = line_actual.mark_point().encode(opacity=alt.condition(selection, alt.value(1), alt.value(0)))
    points_predict = line_predict.mark_point().encode(opacity=alt.condition(selection, alt.value(1), alt.value(0)))
    # 创建竖线图层
    rule = alt.Chart(data).mark_rule(color="gray").encode(
        x="date:T", opacity=alt.condition(selection, alt.value(0.5), alt.value(0)),
        tooltip=[
            alt.Tooltip("date", title="日期"),
            alt.Tooltip("actual_value", title="实际价格"),
            alt.Tooltip("predict_value", title="预测价格"),
        ]).add_selection(selection)
    # 组合所有图层
    chart = (line_actual + line_predict + points_actual + points_predict + rule).interactive()

    return data, chart


@st.cache_data
def convert_df(df):
    return df.to_csv().encode('utf-8')
