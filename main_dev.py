# encoding: utf-8
"""
#@file: main_dev.py
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
from datetime import datetime
import uvicorn
from fastapi import FastAPI, HTTPException
from starlette.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware  # 引入 CORS中间件模块

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

app = FastAPI()
# 设置允许访问的域名
origins = ["*"]  # 也可以设置为"*"，即为所有。

# 设置跨域传参
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # 设置允许的origins来源
    allow_credentials=True,
    allow_methods=["*"],  # 设置允许跨域的http方法，比如 get、post、put等。
    allow_headers=["*"])  # 允许跨域的headers，可以用来鉴别来源等作用。


def get_data(data_type='heat'):
    dl = DataLoader('mysql')
    humiture_outdoor = dl.load(host='8.141.169.219', user='root', password='Zrhdb#2019',
                               port=3307, database='WeiCloudAirDB.V4', sql=Sql.HUMITURE_OUTDOOR.value)
    humiture_indoor = dl.load(host='8.141.169.219', user='root', password='Zrhdb#2019',
                              port=3307, database='WeiCloudAirDB.V4', sql=Sql.HUMITURE_INDOOR.value)
    if data_type == 'heat':
        label = dl.load(host='8.141.169.219', user='root', password='Zrhdb#2019',
                             port=3307, database='WeiCloudAirDB.V4', sql=Sql.HEATER_METER.value)
    elif data_type == 'cooling_tower':
        label = dl.load(host='8.141.169.219', user='root', password='Zrhdb#2019',
                        port=3307, database='WeiCloudAirDB.V4', sql=Sql.COOLING_TOWER.value)
    else:
        label = dl.load(host='8.141.169.219', user='root', password='Zrhdb#2019',
                        port=3307, database='WeiCloudAirDB.V4', sql=Sql.COOLING_PUMP.value)

    # 获取数据
    data_obj = DataPrepare()
    # 预处理，得到特征和标签(可能存在nan值，在model_input中处理)
    x_df = data_obj.build_feature(humiture_outdoor, humiture_indoor)
    y_df = data_obj.build_label(label, data_type=data_type)
    return x_df, y_df


def train(day_data, hour_data=None, retrain=False, save_model=True, model_suffix=''):
    # 日粒度模型训练
    day_df, features_day_df, labels_day_df, features_day_idx = day_data
    print('日粒度可用数据', features_day_df.shape)
    x_train_day, x_test_day, y_train_day, y_test_day = train_test_split(
        features_day_df, labels_day_df, test_size=0.1, random_state=666)
    # 模型训练
    xgb = XgbModel()
    # 尝试加载已有模型
    if model_suffix != '':
        model_suffix = f'_{model_suffix}'
    day_model_path = os.path.join('model_dev', f'xgb_model_day{model_suffix}.json')
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
    plt.figure(figsize=(15, 10))
    plt.plot(y_train_day.values, label='true_train')
    plt.plot(pred_train, label='pred_train')
    plt.figure(figsize=(15, 10))
    plt.plot(y_test_day.values, label='true_test')
    plt.plot(pred, label='pred_test')
    plt.legend()
    plt.show()

    # 小时粒度模型训练
    if hour_data is not None:
        hour_df, features_hour_df, labels_hour_df, features_hour_idx = hour_data
        print('小时粒度可用数据', features_hour_df.shape)
        x_train_hour, x_test_hour, y_train_hour, y_test_hour = train_test_split(
            features_hour_df, labels_hour_df, test_size=0.1, random_state=666)
        # 模型训练
        xgb = XgbModel()
        # 尝试加载已有模型
        hour_model_path = os.path.join('model_dev', f'xgb_model_hour{model_suffix}.json')
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
            sample_model_name = f'kdm_sample_{h}{model_suffix}.model'
            if not retrain:
                km.load(load_path=os.path.join('model_dev', sample_model_name))
            if km.get_model() is None:
                km.fit(ratio_arr)
            if save_model:
                km.save(save_path=os.path.join('model_dev', sample_model_name))
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
    print('day shape', features_day_df.shape)
    # 计算结果进行推导
    xgb = XgbModel()
    # 尝试加载已有模型
    day_model_path = os.path.join('model_dev', 'xgb_model_day.json')
    xgb.load(load_path=day_model_path)
    pred_day_mean = xgb.predict(features_day_df.values)
    valid_day_df = day_df.loc[features_day_idx]
    valid_day_df['pre_capacity_per_day'] = pred_day_mean
    valid_day_df.rename(columns={'date': 'day', 'capacity': 'capacity_per_day'}, inplace=True)
    hour_df, features_hour_df, labels_hour_df, features_hour_idx = hour_data
    print('hour shape', features_hour_df.shape)
    xgb = XgbModel()
    hour_model_path = os.path.join('model_dev', 'xgb_model_hour.json')
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
        sample_model_name = os.path.join('model_dev', f'kdm_sample_{h}.model')
        km.load(load_path=sample_model_name)
        sample_model_dict[h] = km.sample(10).mean()

    predict_df['ratio_sample'] = predict_df['date'].apply(lambda d: sample_model_dict[d.hour])
    predict_df['pre_ratio_opt'] = predict_df['pre_ratio'] * 2 / 3 + predict_df['ratio_sample'] / 3
    # predict_df['pre_ratio_opt'] = predict_df['pre_ratio']
    predict_df['pre_capacity'] = predict_df['pre_capacity_per_day'] * predict_df['pre_ratio_opt']
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
    day_model_path = os.path.join('model_dev', 'xgb_model_day.json')
    xgb.load(load_path=day_model_path)
    pred_day_mean = xgb.predict(features_day)
    xgb = XgbModel()
    hour_model_path = os.path.join('model_dev', 'xgb_model_hour.json')
    xgb.load(load_path=hour_model_path)
    pred_ratio = xgb.predict(features_hour)
    # 采样比例
    # 预加载模型
    hour_list = x_df['date'].dt.hour.drop_duplicates().values
    sample_model_dict = {}
    for h in hour_list:
        km = KDMSample()
        sample_model_name = os.path.join('model_dev', f'kdm_sample_{h}.model')
        km.load(load_path=sample_model_name)
        if km.kde_sample is None:
            sample_model_dict[h] = 0
        else:
            sample_model_dict[h] = km.sample(10).mean()

    ratio_sample = np.array([sample_model_dict[d.hour] for d in x_df['date']])
    pre_ratio_opt = pred_ratio * 2 / 3 + ratio_sample / 3
    res = x_df[['date']].copy()
    res['pre_capacity'] = pred_day_mean * pre_ratio_opt
    res.index = np.arange(res.shape[0])
    return res


def main(data_type='heat'):
    x, y = get_data(data_type=data_type)
    data_obj = DataPrepare()
    if data_type == 'heat':
        day_data = data_obj.model_input_train(x, y, granularity='day')
        hour_data = data_obj.model_input_train(x, y, granularity='hour')
        train(day_data, hour_data)
        val(x, y)
        forecast_per_hour()
    else:
        model_suffix = 'v2' if data_type == 'cooling_tower' else 'v3'
        day_data = data_obj.model_input_train_c(x, y)
        train(day_data, model_suffix=model_suffix)
        forecast_per_day(data_type=data_type)
        day_df, features_day_df, labels_day_df, features_day_idx = day_data
        # 预测结果写入mysql
        xgb = XgbModel()
        # # 尝试加载已有模型
        day_model_path = os.path.join('model_dev', f'xgb_model_day_{model_suffix}.json')
        xgb.load(load_path=day_model_path)
        pre_history = xgb.predict(features_day_df.values)
        pre_res = day_df.loc[features_day_idx][['date']].copy()
        pre_res['predict'] = pre_history
        pre_res = pre_res[(pre_res['date'] >= '2022-06-15') & (pre_res['date'] <= '2022-09-15')]
        res_to_mysql(pre_res, table_name='Tb_Ammeter_Predict', data_type=data_type, interval='1d')
        pre_future = forecast_per_day(data_type=data_type)
        pre_future = pre_future[pre_future['date'] <= '2022-09-15']
        res_to_mysql(pre_future, table_name='Tb_Ammeter_Predict', data_type=data_type, interval='1d')


def forecast_per_hour():
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
    # return pre
    pre['hour'] = pre['date'].dt.hour
    pre['hour'] = pre['hour'].astype(str)
    pre.plot(x='hour', y='pre_capacity', figsize=(15, 10))
    plt.show()


@app.get("/predict", tags=["get load prediction"])
def forecast_per_day(data_type='cooling_tower'):
    try:
        request_params = {'location': '116.298,39.9593', 'key': 'f72973eab37748f5a19b4989ee1466b4'}
        weather_base_url = 'https://api.qweather.com/v7/weather/30d'
        res = requests.get(url=weather_base_url, params=request_params)
        res_dict = json.loads(res.text)
        weather_df = pd.DataFrame(res_dict['daily'])
        feature_df = weather_df[['fxDate', 'tempMin', 'tempMax', 'humidity']].rename(
            columns={'fxDate': 'date', 'tempMin': 't_min', 'tempMax': 't_max', 'humidity': 'h_mean'})
        # 时间需要加8个小时
        feature_df['date'] = pd.to_datetime(feature_df['date']) + pd.Timedelta(hours=8)
        feature_df[['t_min', 't_max', 'h_mean']] = feature_df[['t_min', 't_max', 'h_mean']].astype(float)
        # 适当放大最低温度(靠近工作时段)
        feature_df['t_min'] *= 1.09
        # 室内温度设置为期望温度
        feature_df['sn_t_mean'] = 25
        feature = feature_df[['t_max', 't_min', 'h_mean', 'sn_t_mean']].values
        # 计算结果进行推导
        xgb = XgbModel()
        # 尝试加载已有模型
        model_suffix = 'v2' if data_type == 'cooling_tower' else 'v3'
        day_model_path = os.path.join('model_dev', f'xgb_model_day_{model_suffix}.json')
        xgb.load(load_path=day_model_path)
        pre = xgb.predict(feature)
        pre_df = feature_df[['date']].copy()
        pre_df['predict'] = pre
        res_to_mysql(pre_df, table_name='Tb_Ammeter_Predict_test', data_type=data_type, interval='1d')
        # return pre_df
        # pre_df['day'] = pre_df['date'].dt.strftime('%Y-%m-%d')
        # pre_df.plot(x='day', y='predict', figsize=(15, 10))
        # plt.show()
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        result = JSONResponse(status_code=200, content={"time": current_time, 'result': 'success'})
    except ValueError:
        result = HTTPException(status_code=404, detail="error")
    return result


def res_to_mysql(df, table_name='', project_id=0, data_type='', interval='1H'):
    df['project_id'] = project_id
    if data_type != '':
        df['data_type'] = data_type
    df['current_date'] = pd.Timestamp(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    df.rename(columns={'date': 'pre_date'}, inplace=True)
    df['interval'] = interval
    columns = ['project_id', 'data_type', 'current_date', 'pre_date', 'predict', 'interval']
    df = df[columns]
    DataLoader.pandas_to_sql(df, host='8.141.169.219', user='root', password='Zrhdb#2019',
                             port=3307, database='WeiCloudAirDB.V4', table_name=table_name)


if __name__ == '__main__':
    uvicorn.run(app='main:app', host="0.0.0.0", port=8088, debug=True)
    # main(data_type='cooling_tower')
