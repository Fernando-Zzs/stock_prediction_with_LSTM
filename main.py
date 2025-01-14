# -*- coding: UTF-8 -*-

import logging
import os
import sys
import time
from logging.handlers import RotatingFileHandler

import matplotlib.pyplot as plt
import numpy as np

from data.data_generator import Data

frame = "pytorch"
if frame == "pytorch":
    from model.model_pytorch import train, predict
elif frame == "tensorflow":
    from model.model_tensorflow import train, predict

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'  # tf下会有很多tf的warning，但不影响训练
else:
    raise Exception("Wrong frame seletion")


class Config:
    # 数据参数
    stock_code = "000001"
    start_date = "20170301"
    end_date = "20230411"
    fq_method = "qfq"
    period = "daily"

    all_columns = list(range(0, 29))
    feature_columns = list(range(1, 29))  # 要作为feature的列
    label_columns = [3]  # 要预测的列
    # label_in_feature_index = [feature_columns.index(i) for i in label_columns]  # 这样写不行因为feature不一定从0开始
    label_in_feature_index = (lambda x, y: [x.index(i) for i in y])(feature_columns, label_columns)

    predict_day = 1  # 预测未来几天

    # 网络参数
    input_size = len(feature_columns)
    output_size = len(label_columns)

    hidden_size = 16  # LSTM的隐藏层大小，也是输出大小
    lstm_layers = 2  # LSTM的堆叠层数
    dropout_rate = 0.1  # dropout概率
    time_step = 15  # 这个参数很重要，是设置用前多少天的数据来预测，也是LSTM的time step数，请保证训练数据量大于它

    # 训练参数
    do_train = True
    do_predict = True
    add_train = False  # 是否载入已有模型参数进行增量训练
    shuffle_train_data = True  # 是否对训练数据做shuffle
    use_cuda = False  # 是否使用GPU训练

    train_data_rate = 0.8  # 训练数据占总体数据比例，测试数据就是 1-train_data_rate
    valid_data_rate = 0.15  # 验证数据占训练数据比例，验证集在训练过程使用，为了做模型和参数选择

    batch_size = 64
    learning_rate = 0.01
    epoch = 20  # 整个训练集被训练多少遍，不考虑早停的前提下
    patience = 5  # 训练多少epoch，验证集没提升就停掉
    random_seed = 42  # 随机种子，保证可复现

    do_continue_train = False  # 每次训练把上一次的final_state作为下一次的init_state，仅用于RNN类型模型，目前仅支持pytorch
    continue_flag = ""  # 但实际效果不佳，可能原因：仅能以 batch_size = 1 训练
    if do_continue_train:
        shuffle_train_data = False
        batch_size = 1
        continue_flag = "continue_"

    # 相关性参数
    do_corr_reduction = True
    corr_threshold = 0.3
    duplicate_threshold = 0.9

    # 训练模式
    debug_mode = True  # 调试模式下，是为了跑通代码，追求快
    debug_num = 500  # 仅用debug_num条数据来调试

    # 框架参数
    used_frame = frame  # 选择的深度学习框架，不同的框架模型保存后缀不一样
    model_postfix = {"pytorch": ".pth", "tensorflow": ".ckpt"}
    model_name = "model_" + continue_flag + used_frame + model_postfix[used_frame]

    # 路径参数
    train_data_path = "./data/stock.xlsx"
    model_save_path = "./checkpoint/" + used_frame + "/"
    figure_save_path = "./figure/"
    log_save_path = "./log/"
    do_log_print_to_screen = True
    do_log_save_to_file = True  # 是否将config和训练过程记录到log
    do_figure_save = False
    do_train_visualized = False  # 训练loss可视化，pytorch用tensorboard，tf用tensorboardX
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)  # makedirs 递归创建目录
    if not os.path.exists(figure_save_path):
        os.mkdir(figure_save_path)
    if do_train and (do_log_save_to_file or do_train_visualized):
        cur_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        log_save_path = log_save_path + cur_time + '_' + used_frame + "/"
        os.makedirs(log_save_path)


def load_logger(config):
    logger = logging.getLogger()
    logger.setLevel(level=logging.DEBUG)

    # StreamHandler
    if config.do_log_print_to_screen:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(level=logging.INFO)
        formatter = logging.Formatter(datefmt='%Y/%m/%d %H:%M:%S',
                                      fmt='[ %(asctime)s ] %(message)s')
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    # FileHandler
    if config.do_log_save_to_file:
        file_handler = RotatingFileHandler(config.log_save_path + "out.log", maxBytes=1024000, backupCount=5)
        file_handler.setLevel(level=logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # 把config信息也记录到log 文件中
        config_dict = {}
        for key in dir(config):
            if not key.startswith("_"):
                config_dict[key] = getattr(config, key)
        config_str = str(config_dict)
        config_list = config_str[1:-1].split(", '")
        config_save_str = "\nConfig:\n" + "\n'".join(config_list)
        logger.info(config_save_str)

    return logger


def tidy(config: Config, origin_data: Data, logger, predict_norm_data: np.ndarray):
    label_data = origin_data.data[origin_data.train_num + origin_data.start_num_in_test:,
                 config.label_in_feature_index]
    # predict_data = predict_norm_data * origin_data.std[config.label_in_feature_index] + \
    #                origin_data.mean[config.label_in_feature_index]  # 通过保存的均值和方差还原数据
    predict_data = predict_norm_data * (origin_data.max_val[config.label_in_feature_index] - origin_data.min_val[
        config.label_in_feature_index]) + origin_data.min_val[config.label_in_feature_index]
    assert label_data.shape[0] == predict_data.shape[0], "The element number in origin and predicted data is different"

    label_name = [origin_data.data_column_name[i] for i in config.label_in_feature_index]

    # label 和 predict 是错开config.predict_day天的数据的
    # 下面是两种norm后的loss的计算方式，结果是一样的，可以简单手推一下
    # label_norm_data = origin_data.norm_data[origin_data.train_num + origin_data.start_num_in_test:,
    #              config.label_in_feature_index]
    # loss_norm = np.mean((label_norm_data[config.predict_day:] - predict_norm_data[:-config.predict_day]) ** 2, axis=0)
    # logger.info("The mean squared error of stock {} is ".format(label_name) + str(loss_norm))

    loss = np.mean((label_data[config.predict_day:] - predict_data[:-config.predict_day]) ** 2, axis=0)
    norm_factor = (origin_data.max_val[config.label_in_feature_index] - origin_data.min_val[
        config.label_in_feature_index]) ** 2
    loss_norm = loss * norm_factor
    logger.info("The mean squared error of stock {} is ".format(label_name) + str(loss_norm))

    label_X = range(origin_data.data_num - origin_data.train_num - origin_data.start_num_in_test)
    predict_X = [x + config.predict_day for x in label_X]
    date_df = origin_data.date[-len(label_X):]

    return label_X, label_data, predict_X, predict_data, date_df


def draw(config: Config, origin_data: Data, logger, predict_norm_data: np.ndarray):
    label_column_num = len(config.label_columns)
    label_name = [origin_data.data_column_name[i] for i in config.label_in_feature_index]
    label_X, label_data, predict_X, predict_data, _ = tidy(config, origin_data, logger, predict_norm_data)

    plot_list = []  # 存储所有预测图的列表
    if not sys.platform.startswith('linux'):  # 无桌面的Linux下无法输出，如果是有桌面的Linux，如Ubuntu，可去掉这一行
        for i in range(label_column_num):
            fig = plt.figure(i + 1)  # 预测数据绘制
            plt.plot(label_X, label_data[:, i], label='label')
            plt.plot(predict_X, predict_data[:, i], label='predict')
            plt.legend()
            plt.title("Predict stock {} price with {}".format(label_name[i], config.used_frame))
            logger.info("The predicted stock {} for the next {} day(s) is: ".format(label_name[i], config.predict_day) +
                        str(np.squeeze(predict_data[-config.predict_day:, i])))
            if config.do_figure_save:
                plt.savefig(
                    config.figure_save_path + "{}predict_{}_with_{}.png".format(config.continue_flag, label_name[i],
                                                                                config.used_frame))
            plot_list.append(fig)  # 添加预测图到列表中
        plt.show()
    return plot_list


def main(config):
    logger = load_logger(config)
    try:
        np.random.seed(config.random_seed)  # 设置随机种子，保证可复现
        data_gainer = Data(config)

        if config.do_train:
            train_X, valid_X, train_Y, valid_Y = data_gainer.get_train_and_valid_data()
            train(config, logger, [train_X, train_Y, valid_X, valid_Y])

        if config.do_predict:
            test_X, test_Y = data_gainer.get_test_data(return_label_data=True)
            pred_result = predict(config, test_X)  # 这里输出的是未还原的归一化预测数据
            draw(config, data_gainer, logger, pred_result)
    except Exception:
        logger.error("Run Error", exc_info=True)


if __name__ == "__main__":
    import argparse

    # argparse方便于命令行下输入参数，可以根据需要增加更多
    parser = argparse.ArgumentParser()
    # parser.add_argument("-t", "--do_train", default=False, type=bool, help="whether to train")
    # parser.add_argument("-p", "--do_predict", default=True, type=bool, help="whether to train")
    # parser.add_argument("-b", "--batch_size", default=64, type=int, help="batch size")
    # parser.add_argument("-e", "--epoch", default=20, type=int, help="epochs num")
    args = parser.parse_args()

    con = Config()
    for key in dir(args):  # dir(args) 函数获得args所有的属性
        if not key.startswith("_"):  # 去掉 args 自带属性，比如__name__等
            setattr(con, key, getattr(args, key))  # 将属性值赋给Config

    main(con)
