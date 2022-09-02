# encoding: utf-8
"""
#@file: data.py
#@time: 2022-08-23 14:46
#@author: ywtai 
#@contact: 632910913@qq.com
#@software: PyCharm
#@desc:
"""
from chinese_calendar import is_workday
from collections import defaultdict
# from sklearn.neighbors import LocalOutlierFactor as LOF
from enum import Enum
import os
import pymysql
import numpy as np
import pandas as pd
import re


class Sql(Enum):
    HEATER_METER = "SELECT* FROM Tb_HeatMeter_History WHERE parm_002 IN (469951342248480,469951350586400) " \
                   "and parm_003 BETWEEN '2021-01-01' and '2022-08-30'"
    HUMITURE_OUTDOOR = "SELECT* FROM Tb_TempHumSensor_History WHERE parm_002=470003957458464 " \
                       "AND parm_003>='2021-01-01'AND parm_003<='2022-08-30'"
    HUMITURE_INDOOR = "SELECT* FROM Tb_NewWindController_History WHERE `parm_002` IN " \
                      "(470618646962208,470618572902944,470618744251937,470618803815968," \
                      "470618817306144,470618829469216,470618850919456,470618507798560," \
                      "470618631240224,470618617578528,470618603848736,470618590059040," \
                      "470618560459808,470618546182688,470618530813472) " \
                      "AND `parm_003`<'2022-08-30' AND `parm_003`>='2021-01-01'"


class DataFilter:

    @staticmethod
    def filter_by_3sigma(arr, sig=3):
        temp = arr[(~np.isnan(arr))]
        std = temp.std()
        m = temp.mean()
        use_idx = np.argwhere(((arr >= m - sig * std) & (arr <= m + sig * std)) | np.isnan(arr)).flatten()
        return use_idx


class DataLoader:

    def __init__(self, loader='mysql'):
        self.loader = loader
        self.cache_folder = 'cache'

    def load(self, host='127.0.0.1', user='root', password='', port=3306,
             database='', sql='', cache=True, csv_path='', excel_path='', sheet_name=''):
        if self.loader == 'mysql':
            sql_split = re.split(r' +', sql)
            table_name = sql_split[sql_split.index('FROM') + 1]
            cache_file_name = f'{host}_{user}_{port}_{database}_{table_name}.csv'
            cache_path = os.path.join(self.cache_folder, cache_file_name)
            if cache and os.path.exists(cache_path):
                res_df = pd.read_csv(cache_path)
            else:
                db = pymysql.connect(host=host, user=user, password=password, port=port, database=database)
                cursor = db.cursor()
                cursor.execute(sql)
                results = cursor.fetchall()
                db.close()
                # 根据获取结果生成dataframe
                col = [f[0] for f in cursor.description]
                res_df = pd.DataFrame(list(results), columns=col)
                if cache:
                    if not os.path.exists(self.cache_folder):
                        os.makedirs(self.cache_folder)
                    res_df.to_csv(cache_path, index=False, encoding='utf_8_sig')
        elif self.loader == 'csv':
            res_df = pd.read_csv(csv_path)
        elif self.loader == 'excel':
            res_df = pd.read_excel(excel_path, sheet_name=sheet_name)
        else:
            res_df = None
        return res_df


class DataPrepare:

    @staticmethod
    def build_label(heat_meter_df):
        use_df = heat_meter_df[['parm_002', 'parm_003', 'Hm_Parm_002']]
        use_df = use_df[~pd.isnull(use_df['Hm_Parm_002'])]
        use_df.rename(columns={'parm_002': 'id', 'parm_003': 'date', 'Hm_Parm_002': 'capacity'}, inplace=True)
        use_df['date'] = pd.to_datetime(use_df['date'])
        # 过滤周末和节假日
        use_df['is_workday'] = use_df['date'].apply(lambda x: is_workday(x.to_pydatetime()))
        use_df = use_df.loc[use_df['is_workday']]
        use_df = use_df.drop(columns=['is_workday'])
        # 时间取整
        use_df['date'] = use_df['date'].dt.round('5min')
        # 不同表总量加和
        g = use_df.groupby(['id'])
        all_df = pd.DataFrame(columns=['date', 'capacity'])
        for i, sdf in g:
            sdf = sdf[['date', 'capacity']].sort_values(by=['date'])
            if all_df.empty:
                all_df = sdf.copy()
            else:
                sdf.rename(columns={'capacity': 'temp'}, inplace=True)
                all_df = pd.merge(all_df, sdf, on='date')
                all_df['capacity'] = all_df.apply(lambda s: s['capacity'] + s['temp'], axis=1)
                all_df.drop(columns='temp', inplace=True)
        # lof过滤读数异常点
        # features = all_df['capacity'].values.reshape(-1, 1)
        # lm = LOF(n_neighbors=2)
        # lof_res = lm.fit_predict(features)
        # all_df = all_df.loc[lof_res == 1]
        # 确保时间连续
        temp_df = all_df.copy()
        new_df = pd.DataFrame(pd.date_range(start=temp_df['date'].min(), end=temp_df['date'].max(), freq='5min'),
                              columns=['date'])
        new_df = pd.merge(new_df, temp_df, on='date', how='left')
        new_df = new_df.sort_values(by=['date'])
        new_df['capacity_diff'] = new_df['capacity'].diff()
        # 删除差分为负的点
        new_df.loc[new_df['capacity_diff'] < 0, ['capacity', 'capacity_diff']] = np.nan
        print('start heat meter data filtering')
        print('filter before: ', new_df.shape[0])
        iter_max_counts = 10
        bf = new_df.shape[0]
        for i in range(iter_max_counts):
            diff_use_idx = DataFilter.filter_by_3sigma(new_df['capacity_diff'].values)
            filter_idx = np.setdiff1d(np.arange(bf), diff_use_idx)
            new_df.iloc[filter_idx, [1, 2]] = np.nan
            print('filter: ', len(filter_idx))
            if len(filter_idx) == 0:
                break
        # 过滤非工作时段
        label_df = new_df.copy()
        label_df = label_df[label_df['date'].dt.hour.isin(np.arange(7, 19))]
        label_df = label_df[label_df['date'].dt.month.isin([6, 7, 8])]
        label_df['day'] = label_df['date'].dt.strftime('%Y-%m-%d')
        label_df['hour'] = label_df['date'].dt.hour
        label_df.index = np.arange(len(label_df))
        return label_df

    @staticmethod
    def build_feature(humiture_outdoor_df, humiture_indoor_df=None):
        # 室外温湿度数据处理
        weather_outdoor = humiture_outdoor_df.loc[:, ['parm_003', 'Ths_Parm_001', 'Ths_Parm_002']]
        weather_outdoor.rename(
            columns={'parm_003': 'date', 'Ths_Parm_001': 'temperature',
                     'Ths_Parm_002': 'humidity'}, inplace=True)
        # 湿度处理（数据库中的记录为1-humidity）
        weather_outdoor['humidity'] = 1 - weather_outdoor['humidity']
        weather_outdoor['date'] = pd.to_datetime(weather_outdoor['date'])
        weather_outdoor['date'] = weather_outdoor['date'].dt.round('5min')
        # 确保温湿度时间连续
        supple_dates = pd.DataFrame(
            pd.date_range(start=weather_outdoor['date'].min(), end=weather_outdoor['date'].max(), freq='5min'),
            columns=['date'])
        weather_outdoor = pd.merge(supple_dates, weather_outdoor, on='date', how='left')
        feature_df = weather_outdoor.sort_values(by=['date'])

        # 室内温湿度数据处理
        if humiture_indoor_df is not None:
            weather_indoor = humiture_indoor_df[['parm_002', 'parm_003', 'Nc_parm_002', 'Nc_parm_003']]
            weather_indoor.rename(columns={'parm_002': 'id', 'parm_003': 'date',
                                           'Nc_parm_002': 'temperature_in', 'Nc_parm_003': 'setting_t'}, inplace=True)
            weather_indoor['date'] = pd.to_datetime(weather_indoor['date'])
            # 取室温大于20度的数据
            weather_indoor = weather_indoor[weather_indoor['temperature_in'] >= 20]
            weather_indoor['date'] = weather_indoor['date'].dt.round('10min')
            supple_dates = pd.DataFrame(
                pd.date_range(start=weather_indoor['date'].min(), end=weather_indoor['date'].max(), freq='5min'),
                columns=['date'])
            weather_indoor = pd.merge(supple_dates, weather_indoor, on='date', how='left')
            weather_indoor = weather_indoor.sort_values(by=['date'])
            # 室外室内时间对齐，数据整合
            weather_indoor = weather_indoor[['date', 'temperature_in']].groupby(['date']).mean()
            feature_df = pd.merge(feature_df, weather_indoor, on='date', how='left')

        return feature_df

    def model_input_train(self, x_df, y_df, granularity='day'):
        features, labels = [], []
        features_df, labels_df = pd.DataFrame(), pd.DataFrame()
        feature_idx = []
        merge_df = None
        if granularity == 'day':
            weather_count_thres = 12
            sn_count_thres = 6
            merge_df = self.merge(y_df, step=granularity)
            for idx, s in merge_df.iterrows():
                date_s = s['date'] - pd.Timedelta(days=1)
                date_e = s['date']

                raw_df = x_df[(x_df['date'] >= date_s) & (x_df['date'] <= date_e)]
                # 去掉头和尾的nan值
                first_idx = raw_df[['temperature', 'humidity']].first_valid_index()
                last_idx = raw_df[['temperature', 'humidity']].last_valid_index()
                if first_idx is None or last_idx is None:
                    continue
                ws_df = raw_df.loc[first_idx: last_idx]

                first_idx = raw_df['temperature_in'].first_valid_index()
                last_idx = raw_df['temperature_in'].last_valid_index()
                if first_idx is None or last_idx is None:
                    continue
                in_df = raw_df.loc[first_idx: last_idx]

                if (ws_df[['temperature', 'humidity']].shape[0] < weather_count_thres
                        or in_df['temperature_in'].shape[0] < sn_count_thres):
                    continue

                # 中间的nan值进行插值
                ws_df[['temperature', 'humidity', 'temperature_in']] = ws_df[
                    ['temperature', 'humidity', 'temperature_in']].interpolate(
                    kind='spline', order=3, limit_direction='both')
                # 特征提取
                temperature = ws_df['temperature'].values
                humidity = ws_df['humidity'].values
                temperature_in = ws_df['temperature_in'].values
                # 提取原值、一阶导和二阶导的均值和标准差
                temperature_diff = np.diff(temperature)
                temperature_diff2 = np.diff(temperature_diff)
                humidity_diff = np.diff(humidity)
                humidity_diff2 = np.diff(humidity_diff)
                feature_sub = [temperature.mean(), temperature.std(),
                               temperature_diff.mean(), temperature_diff.std(),
                               temperature_diff2.mean(), temperature_diff2.std(),
                               humidity.mean(), humidity.std(),
                               humidity_diff.mean(), humidity_diff.std(),
                               humidity_diff2.mean(), humidity_diff2.std(), temperature_in.mean()]
                features.append(feature_sub)
                labels.append(s['capacity'] / 12)
                feature_idx.append(idx)

            # features和labels转成dataframe，以便获取索引
            feature_columns = ['t_mean', 't_std', 't_diff_mean', 't_diff_std', 't_diff2_mean', 't_diff2_std',
                               'h_mean', 'h_std', 'h_diff_mean', 'h_diff_std', 'h_diff2_mean', 'h_diff2_std',
                               'sn_t_mean']
            features_df = pd.DataFrame(features, columns=feature_columns)
            labels_df = pd.Series(labels)

        elif granularity == 'hour':
            merge_df = self.merge(y_df, step=granularity)
            merge_day_df = self.merge(y_df)
            merge_df['day'] = merge_df['date'].dt.strftime('%Y-%m-%d')
            merge_df['day'] = pd.to_datetime(merge_df['day'])
            merge_df['hour'] = merge_df['date'].dt.hour

            # 统计每小时占比
            def ratio_cal(se):
                day_cc = merge_day_df[merge_day_df['date'] == se['day']]['capacity'].values
                sub_res = np.nan
                if day_cc.shape[0] > 0:
                    day_cc = day_cc[0]
                    sub_res = se['capacity'] / day_cc * 12
                return sub_res

            merge_df['ratio'] = merge_df.apply(ratio_cal, axis=1)
            merge_df = merge_df[~pd.isnull(merge_df['ratio'])]
            for idx, s in merge_df.iterrows():
                date_s = s['date'] - pd.Timedelta(hours=1)
                date_e = s['date']
                ws_df = x_df[(x_df['date'] >= date_s) & (x_df['date'] <= date_e)].dropna(
                    subset=['temperature', 'humidity'])
                in_df = ws_df.dropna(subset=['temperature_in'])
                if ws_df.shape[0] < 1 or in_df.shape[0] < 1:
                    continue
                # 特征提取
                temperature = ws_df['temperature'].values
                humidity = ws_df['humidity'].values
                sn_temperature = in_df['temperature_in'].values
                feature_sub = [temperature.mean(), humidity.mean(), sn_temperature.mean()]
                # 根据实际条件，只能取均值
                features.append(feature_sub)
                labels.append(s['ratio'])
                feature_idx.append(idx)

            # features和labels转成dataframe，以便获取索引
            feature_columns = ['t_mean', 'h_mean', 'sn_t_mean']
            features_df = pd.DataFrame(features, columns=feature_columns)
            labels_df = pd.Series(labels)

        return merge_df, features_df, labels_df, np.array(feature_idx)

    @staticmethod
    def model_input_predict(x_df):
        """
        处理未来24小时的天气预报数据，构造特征
        :param x_df: 小时级天气预报温湿度数据
        :return:
        """
        use_df = x_df.copy()
        use_df['hour'] = use_df['date'].dt.hour
        use_df = use_df.sort_values(by=['hour'])
        features_day = []
        # 特征提取
        temperature = use_df['temperature'].values
        humidity = use_df['humidity'].values
        temperature_in = use_df['temperature_in'].values
        # 提取原值、一阶导和二阶导的均值和标准差
        temperature_diff = np.diff(temperature)
        temperature_diff2 = np.diff(temperature_diff)
        humidity_diff = np.diff(humidity)
        humidity_diff2 = np.diff(humidity_diff)
        feature_sub = [temperature.mean(), temperature.std(),
                       temperature_diff.mean(), temperature_diff.std(),
                       temperature_diff2.mean(), temperature_diff2.std(),
                       humidity.mean(), humidity.std(),
                       humidity_diff.mean(), humidity_diff.std(),
                       humidity_diff2.mean(), humidity_diff2.std(), temperature_in.mean()]
        features_day.append(feature_sub)
        features_hour = use_df[['temperature', 'humidity', 'temperature_in']].values
        return np.array(features_day), features_hour

    @staticmethod
    def merge(df, step='day'):
        if step == 'day':
            group_list = ['day']
            valid_counts = 144
        else:
            group_list = ['day', 'hour']
            valid_counts = 12

        res_df = defaultdict(list)
        g = df.groupby(group_list)
        for d, sub_df in g:
            if step == 'hour':
                d = pd.Timestamp(d[0]) + pd.Timedelta(hours=d[1])
            else:
                d = pd.Timestamp(d)
            # 去掉开头和结尾的nan值
            first_idx = sub_df['capacity'].first_valid_index()
            last_idx = sub_df['capacity'].last_valid_index()
            if first_idx is None or last_idx is None:
                continue
            sub_df = sub_df.loc[first_idx: last_idx]
            # 计算冷量
            cc = sub_df.iloc[-1]['capacity'] - sub_df.iloc[0]['capacity']
            current_counts = len(sub_df)
            if cc == 0 or current_counts / valid_counts <= 0.5:
                continue
            res_df['date'].append(d)
            res_df['capacity'].append(cc / current_counts * valid_counts)
        return pd.DataFrame.from_dict(res_df)


if __name__ == '__main__':
    dl = DataLoader('mysql')
    heat_meter = dl.load(host='8.141.169.219', user='root', password='Zrhdb#2019',
                         port=3307, database='WeiCloudAirDB.V4', sql=Sql.HEATER_METER.value)
    humiture_outdoor = dl.load(host='8.141.169.219', user='root', password='Zrhdb#2019',
                               port=3307, database='WeiCloudAirDB.V4', sql=Sql.HUMITURE_OUTDOOR.value)
    humiture_indoor = dl.load(host='8.141.169.219', user='root', password='Zrhdb#2019',
                              port=3307, database='WeiCloudAirDB.V4', sql=Sql.HUMITURE_INDOOR.value)
    data_obj = DataPrepare()
    y = data_obj.build_label(heat_meter)
    x = data_obj.build_feature(humiture_outdoor, humiture_indoor)
    print(x.head())
    print(y.head())
    # day_data = data_obj.model_input(x, y, granularity='day')
    # day_df, features_day_df, labels_day_df, features_day_idx = day_data
    # print(day_df)
    # print(features_day_df)
    #
    # hour_data = data_obj.model_input(x, y, granularity='hour')
    # hour_df, features_hour_df, labels_hour_df, features_hour_idx = hour_data
    # print(hour_df)
    # print(features_hour_df)
