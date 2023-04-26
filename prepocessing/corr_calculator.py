# -*- coding = utf-8 -*-
import pandas as pd
import numpy as np


def calculate_corr(df, target_column, corr_threshold=0.2, duplicate_threshold=0.95, random_seed=42):
    # target_column是[int64],需要转化为str
    target_column = df.iloc[:, target_column].columns.tolist()[0]

    # 计算每个指标与目标变量之间的Pearson相关系数得分
    corr_scores = df.corrwith(df[target_column], method='pearson')

    # 将Series转换为DataFrame并重命名列名
    corr_scores_df = pd.DataFrame({'corr_scores': corr_scores}).reset_index()

    # 筛选相关性得分高于corr_threshold的指标
    high_corr_indicators = corr_scores_df[corr_scores_df['corr_scores'].abs() > corr_threshold]['index'].tolist()

    # 记录第一步被筛除掉的列
    dropped_features = corr_scores_df[~corr_scores_df['index'].isin(high_corr_indicators)]['index'].tolist()
    high_corr_indicators.remove(target_column)
    corr_matrix = df[high_corr_indicators].corr(method='pearson').values

    # 设置下三角矩阵掩码
    mask = np.tril(np.ones_like(corr_matrix, dtype=bool), k=-1)

    # 提取大于阈值的元素位置
    idxs = np.argwhere(np.logical_and(mask, corr_matrix >= duplicate_threshold))

    # 随机删除其中的一个特征
    np.random.seed(random_seed)
    for i, j in idxs:
        if high_corr_indicators[i] in dropped_features or high_corr_indicators[j] in dropped_features:
            continue
        dropped_features.append(high_corr_indicators[j] if np.random.randint(2) == 0 else high_corr_indicators[i])

    # 保留剩余的cols
    dropped_cols = set(dropped_features)
    remaining_cols = [col for col in df.columns if col not in dropped_cols]

    return df[remaining_cols], target_column
