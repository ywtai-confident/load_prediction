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
import re
from model import XgbModel, KDMSample
from data import DataLoader, DataPrepare, DataFilter, ArgFactory
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from datetime import datetime
import uvicorn
from fastapi import FastAPI, HTTPException
from starlette.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware  # 引入 CORS中间件模块
from apscheduler.schedulers.background import BackgroundScheduler

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


def get_data(data_type='cold', cache=True):
    dl = DataLoader('mysql')
    parser = ArgFactory()
    humiture_indoor = dl.load(host=parser.con['host'], user=parser.con['user'], password=parser.con['password'],
                              port=parser.con['port'], database=parser.con['database'],
                              sql=parser.sql['humiture_indoor'], cache=cache)
    humiture_outdoor = dl.load(host=parser.con['host'], user=parser.con['user'], password=parser.con['password'],
                               port=parser.con['port'], database=parser.con['database'],
                               sql=parser.sql['humiture_outdoor'], cache=cache)
    if data_type in ['heat', 'cold']:
        label = dl.load(host=parser.con['host'], user=parser.con['user'], password=parser.con['password'],
                        port=parser.con['port'], database=parser.con['database'],
                        sql=parser.sql['heater_meter'], cache=cache)
    elif data_type == 'cooling_tower':
        label = dl.load(host=parser.con['host'], user=parser.con['user'], password=parser.con['password'],
                        port=parser.con['port'], database=parser.con['database'],
                        sql=parser.sql['cooling_tower'], cache=cache)
    else:
        label = dl.load(host=parser.con['host'], user=parser.con['user'], password=parser.con['password'],
                        port=parser.con['port'], database=parser.con['database'],
                        sql=parser.sql['cooling_pump'], cache=cache)

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
    day_model_path = os.path.join('model', f'xgb_model_day{model_suffix}.json')
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
    if hour_data is not None:
        hour_df, features_hour_df, labels_hour_df, features_hour_idx = hour_data
        print('小时粒度可用数据', features_hour_df.shape)
        x_train_hour, x_test_hour, y_train_hour, y_test_hour = train_test_split(
            features_hour_df, labels_hour_df, test_size=0.1, random_state=666)
        # 模型训练
        xgb = XgbModel()
        # 尝试加载已有模型
        hour_model_path = os.path.join('model', f'xgb_model_hour{model_suffix}.json')
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
                km.load(load_path=os.path.join('model', sample_model_name))
            if km.get_model() is None:
                km.fit(ratio_arr)
            if save_model:
                km.save(save_path=os.path.join('model', sample_model_name))
            current_sample = km.sample(2)
            print(f'{h} samples: {current_sample}')


def res_to_mysql(df, table_name='', project_id=0, data_type='', interval='1H', if_exists='append'):
    df['project_id'] = project_id
    if data_type != '':
        df['data_type'] = data_type
    df['current_date'] = pd.Timestamp(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    df.rename(columns={'date': 'pre_date'}, inplace=True)
    df['interval'] = interval
    columns = ['project_id', 'data_type', 'current_date', 'pre_date', 'predict', 'interval']
    df = df[columns]
    df['predict'] = df['predict'].round(4)
    parser = ArgFactory()
    DataLoader.pandas_to_sql(df, host=parser.con['host'], user=parser.con['user'], password=parser.con['password'],
                             port=parser.con['port'], database=parser.con['database'],
                             table_name=table_name, if_exists=if_exists)


@app.get("/train", tags=["training data"])
def train_entry(data_type='cooling_tower'):
    try:
        x, y = get_data(data_type=data_type, cache=False)
        data_obj = DataPrepare()
        # 获取当前本地生成的模型
        parser = ArgFactory()
        max_model_counts = parser.model['max_model_counts']
        model_list = os.listdir('model')
        if data_type in ['cold', 'heat']:
            day_data = data_obj.model_input_train(x, y, granularity='day')
            hour_data = data_obj.model_input_train(x, y, granularity='hour')
            for gran in ['day', 'hour']:
                all_model_list = [re.search(fr'xgb_model_{gran}_{data_type}(.*?)(\d*)\.json', x) for x in model_list]
                use_model_list = np.array([x.group() for x in all_model_list if x is not None])
                model_suffix_list = np.array([int(x.group(2)) for x in all_model_list if x is not None])
                supple_suffix = '_v1'
                if len(model_suffix_list) >= 1:
                    sort_idx = np.argsort(model_suffix_list)
                    supple_suffix = f'_v{model_suffix_list[sort_idx][-1] + 1}'
                    # 模型总数量超过指定个数，删除最早的版本
                    if len(model_suffix_list) == max_model_counts:
                        os.remove(os.path.join('model', use_model_list[sort_idx][0]))
            train(day_data, hour_data, model_suffix=data_type + supple_suffix, retrain=True)
        else:
            day_data = data_obj.model_input_train_c(x, y)
            all_model_list = [re.search(fr'xgb_model_day_{data_type}(.*?)(\d*)\.json', x) for x in model_list]
            use_model_list = np.array([x.group() for x in all_model_list if x is not None])
            model_suffix_list = np.array([int(x.group(2)) for x in all_model_list if x is not None])
            supple_suffix = '_v1'
            if len(model_suffix_list) >= 1:
                sort_idx = np.argsort(model_suffix_list)
                supple_suffix = f'_v{model_suffix_list[sort_idx][-1] + 1}'
                # 模型总数量超过指定个数，删除最早的版本
                if len(model_suffix_list) == max_model_counts:
                    os.remove(os.path.join('model', use_model_list[sort_idx][0]))
            train(day_data, model_suffix=data_type + supple_suffix, retrain=True)
        result = JSONResponse(status_code=200, content={'result': 'success'})

    except ValueError:
        print('无法获取数据或数据量不满足训练条件')
        result = HTTPException(status_code=404, detail="can not get data or data counts not satisfied")
    return result


@app.get("/cover", tags=["cover duplicate data"])
def cover():
    try:
        dl = DataLoader('mysql')
        parser = ArgFactory()
        predict = dl.load(host=parser.con['host'], user=parser.con['user'], password=parser.con['password'],
                          port=parser.con['port'], database=parser.con['database'], sql=parser.sql['predict'], cache=False)
        group = predict.groupby(['project_id', 'data_type', 'pre_date'], sort=False)
        res, beta = [], parser.model['evma_beta']
        for tp, df in group:
            sort_df = df.sort_values(by=['current_date'])
            cd_max = sort_df['current_date'].iloc[-1]
            s = df[df['current_date'] == cd_max].iloc[0]
            s = s.drop('id')
            # predict EVMA(指数加权移动平均)
            if df.shape[0] > 1:
                pre_values = sort_df['predict'].values
                pre_counts = pre_values.shape[0]
                weights = np.logspace(0, pre_counts - 1, pre_counts, base=beta)
                pre_value = np.sum(pre_values * weights * (1 - beta))
                s['predict'] = pre_value
            res.append(s.to_dict())
        res_df = pd.DataFrame(res)
        res_df['predict'] = res_df['predict'].round(4)
        # 重新写入mysql
        DataLoader.pandas_to_sql(res_df, host=parser.con['host'], user=parser.con['user'], password=parser.con['password'],
                                 port=parser.con['port'], database=parser.con['database'],
                                 table_name=parser.table['predict'],
                                 if_exists='replace')
        result = JSONResponse(status_code=200, content={'result': 'success'})
    except ValueError:
        result = HTTPException(status_code=404, detail="cover failed")
    return result


@app.get("/predict", tags=["get load prediction"])
def forecast_per_day(data_type='cooling_tower', project_id=0):
    parser = ArgFactory()
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
        temperature_in = parser.model['temperature_in']
        feature_df['sn_t_mean'] = temperature_in
        feature = feature_df[['t_max', 't_min', 'h_mean', 'sn_t_mean']].values
        # 计算结果进行推导
        xgb = XgbModel()
        # 尝试加载已有模型
        model_list = os.listdir('model')
        all_model_list = [re.search(fr'xgb_model_day_{data_type}(.*?)(\d*)\.json', x) for x in model_list]
        use_model_list = np.array([x.group() for x in all_model_list if x is not None])
        sort_idx = np.argsort([int(x.group(2)) for x in all_model_list if x is not None])
        print('use model:', use_model_list[sort_idx][-1])
        day_model_path = os.path.join('model', use_model_list[sort_idx][-1])
        xgb.load(load_path=day_model_path)
        pre = xgb.predict(feature)
        pre_df = feature_df[['date']].copy()
        pre_df['predict'] = pre
        res_to_mysql(pre_df, table_name=parser.table['predict'], data_type=data_type, interval='1d', project_id=project_id)
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        result = JSONResponse(status_code=200, content={"time": current_time, 'result': 'success'})
    except ValueError:
        result = HTTPException(status_code=404, detail="can not get data or model")
    return result


if __name__ == '__main__':
    # 当model内为空时，会首先执行一次训练
    m_list = os.listdir('model')
    for d in ['cooling_tower', 'cooling_pump']:
        data_list = [x for x in m_list if d in x]
        if not data_list:
            train_entry(d)
    scheduler = BackgroundScheduler(timezone='Asia/Shanghai')
    scheduler.add_job(train_entry, 'cron', day='*/7', hour=1, kwargs={'data_type': 'cooling_tower'})
    scheduler.add_job(train_entry, 'cron', day='*/7', hour=1, kwargs={'data_type': 'cooling_pump'})
    scheduler.add_job(cover, 'cron', day='*/1', hour=1)
    scheduler.start()
    uvicorn.run(app='main:app', host="0.0.0.0", port=8088, debug=True)
