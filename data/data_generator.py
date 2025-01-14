# -*- coding = utf-8 -*-

import akshare as ak
from sklearn.model_selection import train_test_split

from data.database_util import *
from prepocessing.corr_calculator import calculate_corr
from prepocessing.index_calculator import *


class Data:
    def __init__(self, config):
        self.config = config
        self.date, self.data, self.data_column_name = self.get_data()

        self.data_num = self.data.shape[0]
        self.train_num = int(self.data_num * self.config.train_data_rate)

        self.min_val = np.min(self.data, axis=0)
        self.max_val = np.max(self.data, axis=0)
        self.norm_data = (self.data - self.min_val) / (self.max_val - self.min_val + 1e-8)

        self.start_num_in_test = 0  # 测试集中前几天的数据会被删掉，因为它不够一个time_step

    def get_data(self):  # 获取初始数据
        if self.config.debug_mode:
            init_data = get_from_db(self.config.stock_code)
        else:
            init_data = self.get_from_web()
            save_to_db(init_data, self.config.stock_code, False)
        init_data.dropna(inplace=True)

        # 如果删除相关性冗余的预测变量，则需要重置config中的feature_columns;label_columns及其联动项
        if self.config.do_corr_reduction:
            init_data, target_column = calculate_corr(init_data, self.config.label_columns, self.config.corr_threshold,
                                                      self.config.duplicate_threshold)
            self.config.label_columns = [init_data.columns.get_loc(target_column)]
            self.config.feature_columns = list(range(1, init_data.shape[1]))
            self.config.label_in_feature_index = (lambda x, y: [x.index(i) for i in y])(self.config.feature_columns,
                                                                                        self.config.label_columns)
            self.config.input_size = len(self.config.feature_columns)
            self.config.output_size = len(self.config.label_columns)

        return init_data.iloc[:, 0].values, init_data.iloc[:,
                                            self.config.feature_columns].values, init_data.columns.tolist()

    def get_from_web(self):  # 从网上获取数据并计算指标
        df = ak.stock_zh_a_hist(symbol=self.config.stock_code, period=self.config.period,
                                start_date=self.config.start_date, end_date=self.config.end_date,
                                adjust=self.config.fq_method)
        df = df.drop(columns=['涨跌幅', '涨跌额'])
        df.rename(columns={'日期': 'date', '开盘': 'open', '收盘': 'close', '最高': 'high', '最低': 'low', '换手率': 'turnover',
                           '成交量': 'volume', '成交额': 'amount', '振幅': 'amplitude'}, inplace=True)
        df = (df.pipe(calculate_average_line)
              .pipe(calculate_macd)
              .pipe(calculate_kdj)
              .pipe(calculate_dmi)
              .pipe(calculate_bollinger_bands)
              .pipe(calculate_average_true_range))
        return df.iloc[:, self.config.all_columns]

    def get_train_and_valid_data(self):
        feature_data = self.norm_data[:self.train_num]
        label_data = self.norm_data[self.config.predict_day: self.config.predict_day + self.train_num,
                     self.config.label_in_feature_index]  # 将延后几天的数据作为label

        if not self.config.do_continue_train:
            # 在非连续训练模式下，每time_step行数据会作为一个样本，两个样本错开一行，比如：1-20行，2-21行。。。。
            train_x = [feature_data[i:i + self.config.time_step] for i in range(self.train_num - self.config.time_step)]
            train_y = [label_data[i:i + self.config.time_step] for i in range(self.train_num - self.config.time_step)]
        else:
            # 在连续训练模式下，每time_step行数据会作为一个样本，两个样本错开time_step行，
            # 比如：1-20行，21-40行。。。到数据末尾，然后又是 2-21行，22-41行。。。到数据末尾，……
            # 这样才可以把上一个样本的final_state作为下一个样本的init_state，而且不能shuffle
            # 目前本项目中仅能在pytorch的RNN系列模型中用
            train_x = [
                feature_data[start_index + i * self.config.time_step: start_index + (i + 1) * self.config.time_step]
                for start_index in range(self.config.time_step)
                for i in range((self.train_num - start_index) // self.config.time_step)]
            train_y = [
                label_data[start_index + i * self.config.time_step: start_index + (i + 1) * self.config.time_step]
                for start_index in range(self.config.time_step)
                for i in range((self.train_num - start_index) // self.config.time_step)]

        train_x, train_y = np.array(train_x), np.array(train_y)

        train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=self.config.valid_data_rate,
                                                              random_state=self.config.random_seed,
                                                              shuffle=self.config.shuffle_train_data)  # 划分训练和验证集，并打乱
        return train_x, valid_x, train_y, valid_y

    def get_test_data(self, return_label_data=False):
        feature_data = self.norm_data[self.train_num:]
        sample_interval = min(feature_data.shape[0], self.config.time_step)  # 防止time_step大于测试集数量
        self.start_num_in_test = feature_data.shape[0] % sample_interval  # 这些天的数据不够一个sample_interval
        time_step_size = feature_data.shape[0] // sample_interval

        # 在测试数据中，每time_step行数据会作为一个样本，两个样本错开time_step行
        # 比如：1-20行，21-40行。。。到数据末尾。
        test_x = [feature_data[
                  self.start_num_in_test + i * sample_interval: self.start_num_in_test + (i + 1) * sample_interval]
                  for i in range(time_step_size)]
        if return_label_data:  # 实际应用中的测试集是没有label数据的
            label_data = self.norm_data[self.train_num + self.start_num_in_test:, self.config.label_in_feature_index]
            return np.array(test_x), label_data
        return np.array(test_x)
