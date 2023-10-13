# -*- coding: utf-8 -*-
# @Author  : WangYaxi
# @Time    : 2023/10/13 23:43
# @File    : LoadTrainingDataAndLabels.py
# @Software: PyCharm 
# describe : 获取训练数据和标签
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from torch import from_numpy


def load_train_data():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    rawData = pd.read_csv(filepath_or_buffer='../../data/BAI246_CAL_GR_AC_SP.csv', header=0, encoding='utf-8',
                          dtype=np.float32, on_bad_lines='skip')
    data = rawData.values[:50, 1:5]
    labels = rawData.columns[1:]

    # 是否使用归一化数据
    uniformization = True
    if uniformization:
        num_pipeline = Pipeline([('std_scaler', MinMaxScaler())])
        train_data = num_pipeline.fit_transform(data)
    else:
        train_data = data
    return from_numpy(train_data), labels


if __name__ == '__main__':
    train_data, labels = load_train_data()
