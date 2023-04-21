# -*- coding = utf-8 -*-

import importlib
from datetime import datetime

import altair as alt
import numpy as np
import streamlit as st

from data.data_generator import Data
from main import Config, load_logger, tidy, view_options
from prepocessing.date_util import calc_bdate, gap_period
from view.chart_util import get_chart

alt.themes.enable("streamlit")

# 设置应用程序的标题和页眉
st.set_page_config(page_title="Stock Price Predictor", page_icon=":chart_with_upwards_trend:", layout="wide")

# 创建侧边栏，用于用户输入模型参数
st.sidebar.title("自定义预测参数")
config = Config()
st.sidebar.markdown('---')
config.stock_code = st.sidebar.text_input("股票代码", value='000001')
config.start_date = datetime.strftime(
    st.sidebar.date_input("起始日期", value=gap_period(datetime.strftime(datetime.now(), "%Y%m%d"), 4, True),
                          max_value=gap_period(config.end_date, 3, True)), "%Y%m%d")
config.end_date = datetime.strftime(st.sidebar.date_input("结束日期", value=datetime.now(),
                                                          min_value=gap_period(config.start_date, 3, False)), "%Y%m%d")
config.fq_method = view_options[
    st.sidebar.radio("复权类型", ('前复权', '后复权'), horizontal=True, help="选择前复权能够更好地反映短期价格趋势，选择后复权能够更好地反映长期涨跌情况")]
st.sidebar.markdown('---')

with st.sidebar.expander("网络参数"):
    config.used_frame = st.radio("框架", ('PyTorch', 'TensorFlow'), horizontal=True)
    config.predict_day = st.slider("预测天数", min_value=1, max_value=7, step=1, value=1)
    config.hidden_size = st.radio("隐藏层大小", (64, 128, 256), index=1, horizontal=True, help="网络的每层隐藏层的规模")
    config.lstm_layers = st.slider("堆叠层数", min_value=1, max_value=8, step=1, value=2, help="网络的LSTM堆叠层数")
    config.dropout_rate = st.slider("丢包率", min_value=0.0, max_value=1.0, step=0.05, value=0.2,
                                    help="作用于LSTM网络之间以一定概率丢失数据，防止过拟合")
    config.time_step = st.number_input("时间步长", min_value=1,
                                       max_value=calc_bdate(config.start_date, config.end_date) - 33,
                                       step=1, value=20, help="用前多少天的数据来预测，也是LSTM的time step数")

with st.sidebar.expander("训练参数"):
    config.add_train = st.checkbox("增量训练", value=False, help="是否载入已有模型参数进行增量训练")
    config.shuffle_train_data = st.checkbox("打乱训练", value=True, help="是否对训练数据做打乱操作")
    config.use_cuda = st.checkbox("使用GPU训练", value=False)
    config.do_continue_train = st.checkbox("持续训练", value=False, disabled=config.used_frame == 'TensorFlow',
                                           help="每次训练把上一次的final_state作为下一次的init_state")
    config.debug_mode = st.checkbox("离线模式", value=False, help="不联网获取数据，请确保数据库为最新数据")

st.sidebar.markdown('---')
config.label_columns = [view_options[selected_option] for selected_option in
                        st.sidebar.multiselect("预测变量", ['开盘价', '收盘价', '最高价', '最低价'], ['收盘价'])]

# 创建标签页，用于展示结果和数据管理
tab_chart, tab_data = st.tabs([":bar_chart: CHART", ":clipboard: DATA"])
chart_placeholder = tab_chart.empty()
data_placeholder = tab_data.empty()

# TODO:运行前需要检查：do_continue_train变更时同时变更
if st.sidebar.button("运行", type='primary', use_container_width=True):
    # 修改预测列需要联动修改其他参数
    config.label_in_feature_index = (lambda x, y: [x.index(i) for i in y])(config.feature_columns, config.label_columns)
    config.output_size = len(config.label_columns)

    logger = load_logger(config)
    module = None
    if config.used_frame == 'PyTorch':
        module = importlib.import_module("model.model_pytorch")
    else:
        module = importlib.import_module("model.model_tensorflow")
    try:
        np.random.seed(config.random_seed)  # 设置随机种子，保证可复现
        data_gainer = Data(config)

        if config.do_train:
            train_X, valid_X, train_Y, valid_Y = data_gainer.get_train_and_valid_data()
            module.train(config, logger, [train_X, train_Y, valid_X, valid_Y])

        if config.do_predict:
            test_X, test_Y = data_gainer.get_test_data(return_label_data=True)
            pred_result = module.predict(config, test_X)  # 这里输出的是未还原的归一化预测数据
            _, label_data, _, predict_data, date_df = tidy(config, data_gainer, logger, pred_result)
            for i in range(len(config.label_columns)):
                df, chart = get_chart(date_df, label_data[:, i], predict_data[:, i], config.predict_day)
                data_placeholder.dataframe(data=df, use_container_width=True)
                chart_placeholder.altair_chart(chart, use_container_width=True)

    except Exception:
        logger.error("Run Error", exc_info=True)
