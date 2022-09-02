# encoding: utf-8
"""
#@file: main.py
#@time: 2022-08-23 14:45
#@author: ywtai
#@contact: 632910913@qq.com
#@software: PyCharm
#@desc:
"""
import os
import requests
import json
from model import XgbModel, KDMSample
from data import DataLoader, DataPrepare, DataFilter, Sql
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 显示所有列(参数设置为None代表显示所有行，也可以自行设置数字)
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置数据的显示长度，默认为50
pd.set_option('max_colwidth', 200)
# 禁止自动换行(设置为False不自动换行，True反之)
pd.set_option('expand_frame_repr', False)


def get_data():
    dl = DataLoader('mysql')
    heat_meter = dl.load(host='8.141.169.219', user='root', password='Zrhdb#2019',
                         port=3307, database='WeiCloudAirDB.V4', sql=Sql.HEATER_METER.value)
    humiture_outdoor = dl.load(host='8.141.169.219', user='root', password='Zrhdb#2019',
                               port=3307, database='WeiCloudAirDB.V4', sql=Sql.HUMITURE_OUTDOOR.value)
    humiture_indoor = dl.load(host='8.141.169.219', user='root', password='Zrhdb#2019',
                              port=3307, database='WeiCloudAirDB.V4', sql=Sql.HUMITURE_INDOOR.value)
    # 获取数据
    data_obj = DataPrepare()
    # 预处理，得到特征和标签(可能存在nan值，在model_input中处理)
    x_df = data_obj.build_feature(humiture_outdoor, humiture_indoor)
    y_df = data_obj.build_label(heat_meter)
    return x_df, y_df


def train(day_data, hour_data, retrain=False, save_model=True):
    # 日粒度模型训练
    day_df, features_day_df, labels_day_df, features_day_idx = day_data
    print('日粒度可用数据', features_day_df.shape)
    x_train_day, x_test_day, y_train_day, y_test_day = train_test_split(
        features_day_df, labels_day_df, test_size=0.1, random_state=666)
    # 模型训练
    xgb = XgbModel()
    # 尝试加载已有模型
    day_model_path = os.path.join('model', 'xgb_model_day.json')
    if not retrain:
        xgb.load(load_path=day_model_path)
    if xgb.get_model() is None:
        xgb.fit(x_train_day.values, y_train_day.values)
    if save_model:
        xgb.save(save_path=day_model_path)
    # 计算训练集和测试集的mse
    pred = xgb.predict(x_test_day.values)
    pred_train = xgb.predict(x_train_day.values)
    mse_train = np.sqrt(mean_squared_error(y_train_day.values, pred_train))
    mse_test = np.sqrt(mean_squared_error(y_test_day.values, pred))
    print('日粒度rmse（训练集）:', mse_train)
    print('日粒度rmse（测试集）:', mse_test)

    # 小时粒度模型训练
    hour_df, features_hour_df, labels_hour_df, features_hour_idx = hour_data
    print('小时粒度可用数据', features_hour_df.shape)
    x_train_hour, x_test_hour, y_train_hour, y_test_hour = train_test_split(
        features_hour_df, labels_hour_df, test_size=0.1, random_state=666)
    # 模型训练
    xgb = XgbModel()
    # 尝试加载已有模型
    hour_model_path = os.path.join('model', 'xgb_model_hour.json')
    if not retrain:
        xgb.load(load_path=hour_model_path)
    if xgb.get_model() is None:
        xgb.fit(x_train_hour.values, y_train_hour.values)
    if save_model:
        xgb.save(save_path=hour_model_path)
    # 计算训练集和测试集的mse
    pred_hour = xgb.predict(x_test_hour.values)
    pred_train_hour = xgb.predict(x_train_hour.values)
    mse_train = np.sqrt(mean_squared_error(y_train_hour.values, pred_train_hour))
    mse_test = np.sqrt(mean_squared_error(y_test_hour.values, pred_hour))
    print('小时粒度rmse（训练集）:', mse_train)
    print('小时粒度rmse（测试集）:', mse_test)

    # 核密度估计+马尔可夫链+蒙特卡洛采样，计算冷量占比，每小时对应一个采样模型
    hour_group = hour_df.groupby(['hour'])
    for h, df in hour_group:
        ratio_arr = df['ratio'].values
        ratio_arr = ratio_arr[DataFilter.filter_by_3sigma(ratio_arr)]
        km = KDMSample()
        sample_model_name = f'kdm_sample_{h}.model'
        if not retrain:
            km.load(load_path=os.path.join('model', sample_model_name))
        if km.get_model() is None:
            km.fit(ratio_arr)
        if save_model:
            km.save(save_path=os.path.join('model', sample_model_name))
        current_sample = km.sample(2)
        print(f'{h} samples: {current_sample}')


def val(x_df, y_df):
    """

    :param x_df:
    :param y_df:
    :return:
    """

    data_obj = DataPrepare()
    day_data = data_obj.model_input_train(x_df, y_df, granularity='day')
    hour_data = data_obj.model_input_train(x_df, y_df, granularity='hour')
    day_df, features_day_df, labels_day_df, features_day_idx = day_data
    print('load success, continuing')
    print('day shape', day_df.shape)
    # 计算结果进行推导
    xgb = XgbModel()
    # 尝试加载已有模型
    day_model_path = os.path.join('model', 'xgb_model_day.json')
    xgb.load(load_path=day_model_path)
    pred_day_mean = xgb.predict(features_day_df.values)
    valid_day_df = day_df.loc[features_day_idx]
    valid_day_df['pre_capacity_per_day'] = pred_day_mean
    valid_day_df.rename(columns={'date': 'day', 'capacity': 'capacity_per_day'}, inplace=True)
    hour_df, features_hour_df, labels_hour_df, features_hour_idx = hour_data
    print('hour shape', hour_df.shape)
    xgb = XgbModel()
    hour_model_path = os.path.join('model', 'xgb_model_hour.json')
    xgb.load(load_path=hour_model_path)
    pred_ratio = xgb.predict(features_hour_df.values)
    valid_hour_df = hour_df.loc[features_hour_idx]
    valid_hour_df['pre_ratio'] = pred_ratio
    predict_df = pd.merge(valid_hour_df, valid_day_df, on='day')
    print(predict_df.columns)
    print(predict_df.shape)
    # 采样比例
    # 预加载模型
    hour_list = predict_df['date'].dt.hour.drop_duplicates().values
    sample_model_dict = {}
    for h in hour_list:
        km = KDMSample()
        sample_model_name = os.path.join('model', f'kdm_sample_{h}.model')
        km.load(load_path=sample_model_name)
        sample_model_dict[h] = km.sample(10).mean()

    predict_df['ratio_sample'] = predict_df['date'].apply(lambda d: sample_model_dict[d.hour])
    predict_df['pre_ratio_opt'] = predict_df['pre_ratio'] * 2 / 3 + predict_df['ratio_sample'] / 3
    predict_df['pre_capacity'] = predict_df['pre_capacity_per_day'] * predict_df['pre_ratio_opt']
    print(predict_df.head())
    true_y = predict_df['capacity'].values
    pre_y = predict_df['pre_capacity'].values
    plt.figure(figsize=(15, 10))
    plt.plot(true_y, label='true')
    plt.plot(pre_y, label='predict')
    plt.legend()
    plt.show()


def predict(x_df):
    """
    预测未来24小时的冷热量
    :param x_df:
    :return:
    """
    data_obj = DataPrepare()
    features_day, features_hour = data_obj.model_input_predict(x_df)
    # 计算结果进行推导
    xgb = XgbModel()
    # 尝试加载已有模型
    day_model_path = os.path.join('model', 'xgb_model_day.json')
    xgb.load(load_path=day_model_path)
    pred_day_mean = xgb.predict(features_day)
    xgb = XgbModel()
    hour_model_path = os.path.join('model', 'xgb_model_hour.json')
    xgb.load(load_path=hour_model_path)
    pred_ratio = xgb.predict(features_hour)
    # 采样比例
    # 预加载模型
    hour_list = x_df['date'].dt.hour.drop_duplicates().values
    sample_model_dict = {}
    for h in hour_list:
        km = KDMSample()
        sample_model_name = os.path.join('model', f'kdm_sample_{h}.model')
        km.load(load_path=sample_model_name)
        if km.kde_sample is None:
            sample_model_dict[h] = 0
        else:
            sample_model_dict[h] = km.sample(10).mean()

    ratio_sample = np.array([sample_model_dict[d.hour] for d in x_df['date']])
    pre_ratio_opt = pred_ratio * 2 / 3 + ratio_sample / 3
    x_df['pre_capacity'] = pred_day_mean * pre_ratio_opt
    x_df.index = np.arange(x_df.shape[0])
    return x_df[['date', 'pre_capacity']].copy()


def main():
    x, y = get_data()
    data_obj = DataPrepare()
    day_data = data_obj.model_input_train(x, y, granularity='day')
    hour_data = data_obj.model_input_train(x, y, granularity='hour')
    train(day_data, hour_data)
    val(x, y)


def forecast_feature():
    request_params = {'location': '116.298,39.9593', 'key': 'f72973eab37748f5a19b4989ee1466b4'}
    weather_base_url = 'https://api.qweather.com/v7/grid-weather/24h'
    res = requests.get(url=weather_base_url, params=request_params)
    res_dict = json.loads(res.text)
    weather_df = pd.DataFrame(res_dict['hourly'])
    feature_df = weather_df[['fxTime', 'temp', 'humidity']].rename(columns={'fxTime': 'date', 'temp': 'temperature'})
    # 时间需要加8个小时
    feature_df['date'] = pd.to_datetime(feature_df['date']) + pd.Timedelta(hours=8)
    feature_df[['temperature', 'humidity']] = feature_df[['temperature', 'humidity']].astype(float)
    # 室内温度设置为期望温度
    feature_df['temperature_in'] = 25
    pre = predict(feature_df)
    pre['hour'] = pre['date'].dt.hour
    pre['hour'] = pre['hour'].astype(str)
    pre.plot(x='hour', y='pre_capacity', figsize=(15, 10))
    plt.legend()
    plt.show()


if __name__ == '__main__':
    forecast_feature()
