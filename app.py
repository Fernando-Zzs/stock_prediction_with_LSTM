# -*- coding = utf-8 -*-

import importlib
from datetime import datetime

import numpy as np
import streamlit as st
from dateutil.relativedelta import relativedelta

from data.data_generator import Data
from main import Config, load_logger, draw
from utils.date_util import calc_bdate

view_options = {
    '前复权': 'qfq',
    '后复权': 'hfq',
    '开盘价': 1,
    '收盘价': 2,
    '最高价': 3,
    '最低价': 4
}

# 设置应用程序的标题和页眉
st.set_page_config(page_title="Stock Price Predictor", page_icon=":chart_with_upwards_trend:", layout="wide")
st.title("股票价格预测")
# 创建侧边栏，用于用户输入模型参数
st.sidebar.title("自定义预测参数")
config = Config()

config.stock_code = st.sidebar.text_input("股票代码", value='000001')
config.start_date = datetime.strftime(st.sidebar.date_input("起始日期", value=datetime.now() - relativedelta(months=3),
                                                            max_value=datetime.strptime(config.end_date,
                                                                                        "%Y%m%d") - relativedelta(
                                                                months=2)), "%Y%m%d")
config.end_date = datetime.strftime(st.sidebar.date_input("结束日期", value=datetime.now(),
                                                          min_value=datetime.strptime(config.start_date,
                                                                                      "%Y%m%d") + relativedelta(
                                                              months=2)), "%Y%m%d")
config.fq_method = view_options[st.sidebar.radio("复权类型", ('前复权', '后复权'))]

with st.sidebar.expander("网络参数"):
    config.label_columns = [view_options[selected_option] for selected_option in
                            st.sidebar.multiselect("预测项", ['开盘价', '收盘价', '最高价', '最低价'], ['收盘价'])]
    config.used_frame = st.radio("框架", ('PyTorch', 'TensorFlow'), horizontal=True)
    config.predict_day = st.slider("预测天数", min_value=1, max_value=7, step=1, value=1)
    config.hidden_size = st.radio("隐藏层大小", (64, 128, 256), index=1, horizontal=True)
    config.lstm_layers = st.slider("堆叠层数", min_value=1, max_value=8, step=1, value=2)
    config.dropout_rate = st.slider("丢包率", min_value=0.0, max_value=1.0, step=0.05, value=0.2)
    config.time_step = st.number_input("时间步长", min_value=1, max_value=calc_bdate(config.start_date, config.end_date),
                                       step=1, value=20)

# TODO:运行前需要检查：所有空都有填写
if st.sidebar.button("运行"):
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
            plot_list = draw(config, data_gainer, logger, pred_result)
            for fig in plot_list:
                st.pyplot(fig)
    except Exception:
        logger.error("Run Error", exc_info=True)
