# -*- coding = utf-8 -*-

import importlib

import numpy as np
import streamlit as st

from data.data_generator import Data
from main import Config, load_logger, draw

# 设置应用程序的标题和页眉
st.set_page_config(page_title="Stock Price Predictor", page_icon=":chart_with_upwards_trend:", layout="wide")
st.title("股票价格预测")
# 创建侧边栏，用于用户输入模型参数
st.sidebar.title("模型参数")
config = Config()
config.stock_code = st.sidebar.text_input("股票代码", value='000001')
config.start_date = st.sidebar.text_input("开始日期", value='20220101')
config.end_date = st.sidebar.text_input("结束日期", value='20220331')
config.predict_day = st.sidebar.slider("预测天数", min_value=1, max_value=10, step=1, value=1)
framework = st.sidebar.radio("框架", ('PyTorch', 'TensorFlow'))
if st.sidebar.button("运行"):
    logger = load_logger(config)
    module = None
    if framework == 'PyTorch':
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
