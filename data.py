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
import os
import yaml
from sqlalchemy import create_engine
import pymysql
import numpy as np
import pandas as pd
import re
import hashlib


# class lazy_property:
#     """
#     描述器
#     """
#     def __init__(self, func):
#         self.func = func
#
#     def __get__(self, instance, cls):
#         if instance is None:
#             return self
#         else:
#             value = self.func(instance)
#             setattr(instance, self.func.__name__, value)
#             return value
class Singleton(object):
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, '_instance'):
            cls._instance = super(Singleton, cls).__new__(cls, *args, **kwargs)
        return cls._instance


class ArgFactory(Singleton):

    def __init__(self, arg_path=None):
        if arg_path is None:
            arg_path = 'arg.yml'
        with open(arg_path, 'r') as f:
            arg_dict = yaml.load(f.read(), Loader=yaml.FullLoader)
        self.con = arg_dict['con']
        self.table = arg_dict['table']
        self.sql = arg_dict['sql']
        self.model = arg_dict['model']


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
            # sql使用md5编码(在此之前去掉多余的空格)
            sql = re.sub(r' +', ' ', sql)
            sql_encode = hashlib.md5(sql.encode()).hexdigest()
            cache_file_name = f'{host}_{user}_{port}_{database}_{table_name}_{sql_encode}.csv'
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

    @staticmethod
    def pandas_to_sql(df, table_name='predict', host='127.0.0.1',
                      user='root', password='', port=3306, database='', if_exists='append'):

        db_string = f'mysql+pymysql://{user}:{password}@{host}:{port}/{database}'
        engine = create_engine(db_string)
        # 判断表是否存在
        db = pymysql.connect(host=host, user=user, password=password, port=port, database=database)
        cursor = db.cursor()
        cursor.execute("show tables")
        tables = cursor.fetchall()
        db.close()
        table_list = [t[0] for t in tables]
        add_keys = False
        if table_name not in table_list:
            df.to_sql(table_name, engine, index=False)
            add_keys = True
        else:
            df.to_sql(table_name, engine, index=False, if_exists=if_exists)
            if if_exists == 'replace':
                add_keys = True
        if add_keys:
            # 添加自增主键
            with engine.connect() as con:
                con.execute(f"ALTER TABLE`WeiCloudAirDB.V4`.`{table_name}` "
                            f"ADD COLUMN `id` INT NOT NULL AUTO_INCREMENT FIRST, "
                            f"ADD PRIMARY KEY (`id`);")


class DataPrepare:

    @staticmethod
    def build_label(meter_df, data_type='cold'):
        key = 'Hm_Parm_002' if data_type in ['heat', 'cold'] else 'Am_Parm_029'
        use_df = meter_df[['parm_002', 'parm_003', key]]
        use_df = use_df[~pd.isnull(use_df[key])]
        use_df.rename(columns={'parm_002': 'id', 'parm_003': 'date', key: 'capacity'}, inplace=True)
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
        label_df = label_df[label_df['date'].dt.month.isin([6, 7, 8, 9])]
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
        weather_outdoor['humidity'] = 100 - weather_outdoor['humidity']
        weather_outdoor['date'] = pd.to_datetime(weather_outdoor['date'])
        weather_outdoor['date'] = weather_outdoor['date'].dt.round('5min')
        # weather_outdoor = weather_outdoor[(weather_outdoor['temperature'] >= 10)
        #                                   & (weather_outdoor['temperature'] <= 50)]
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
            weather_indoor = weather_indoor[(weather_indoor['temperature_in'] >= 24)
                                            & (weather_indoor['temperature_in'] <= 32)]
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
                # 取工作日时段
                raw_df = raw_df[raw_df['date'].dt.hour.isin(np.arange(7, 19))]
                # 去掉头和尾的nan值
                first_idx = raw_df[['temperature', 'humidity']].first_valid_index()
                last_idx = raw_df[['temperature', 'humidity']].last_valid_index()
                if first_idx is None or last_idx is None:
                    continue
                ws_df = raw_df.loc[first_idx: last_idx].copy()
                in_df = raw_df.dropna(subset=['temperature_in'])

                if (ws_df[['temperature', 'humidity']].shape[0] < weather_count_thres
                        or in_df['temperature_in'].shape[0] < sn_count_thres):
                    continue

                # 中间的nan值进行插值
                ws_df[['temperature', 'humidity']] = ws_df[
                    ['temperature', 'humidity']].interpolate(kind='spline', order=3, limit_direction='both')
                # 特征提取
                temperature = ws_df['temperature'].values
                humidity = ws_df['humidity'].values
                temperature_in = in_df['temperature_in'].values
                # 提取均值和标准差
                feature_sub = [temperature.mean(), temperature.std(),
                               humidity.mean(), humidity.std(),
                               temperature_in.mean()]
                features.append(feature_sub)
                labels.append(s['capacity'] / 12)
                feature_idx.append(idx)

            # features和labels转成dataframe，以便获取索引
            feature_columns = ['t_mean', 't_std', 'h_mean', 'h_std', 'sn_t_mean']
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

    def model_input_train_c(self, x_df, y_df):
        """
        能耗预测构建特征的方式(需要预测未来一个月,无法构建导数等复杂特征)
        :param x_df:
        :param y_df:
        :return:
        """
        features, labels = [], []
        feature_idx = []
        weather_count_thres = 6
        sn_count_thres = 3
        merge_df = self.merge(y_df)
        for idx, s in merge_df.iterrows():
            date_s = s['date'] - pd.Timedelta(days=1)
            date_e = s['date']

            raw_df = x_df[(x_df['date'] >= date_s) & (x_df['date'] <= date_e)]
            raw_df = raw_df[raw_df['date'].dt.hour.isin(np.arange(7, 19))]
            ws_df = raw_df.dropna(subset=['temperature', 'humidity'])
            in_df = raw_df.dropna(subset=['temperature_in'])

            if (ws_df[['temperature', 'humidity']].shape[0] < weather_count_thres
                    or in_df['temperature_in'].shape[0] < sn_count_thres):
                continue

            # 特征提取
            temperature = ws_df['temperature'].values
            humidity = ws_df['humidity'].values
            temperature_in = in_df['temperature_in'].values
            # 提取原值、一阶导和二阶导的均值和标准差
            feature_sub = [temperature.max(), temperature.min(),
                           humidity.mean(), temperature_in.mean()]
            features.append(feature_sub)
            labels.append(s['capacity'])
            feature_idx.append(idx)

        # features和labels转成dataframe，以便获取索引
        feature_columns = ['t_max', 't_min', 'h_mean', 'sn_t_mean']
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
        feature_sub = [temperature.mean(), temperature.std(),
                       humidity.mean(), humidity.std(),
                       temperature_in.mean()]
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
            if cc <= 0 or current_counts / valid_counts <= 0.5:
                continue
            res_df['date'].append(d)
            res_df['capacity'].append(cc / current_counts * valid_counts)
        return pd.DataFrame.from_dict(res_df)


if __name__ == '__main__':
    dl = DataLoader('mysql')
    parser = ArgFactory()
    heat_meter = dl.load(host=parser.con['host'], user=parser.con['user'], password=parser.con['password'],
                         port=parser.con['port'], database=parser.con['database'], sql=parser.sql['heater_meter'])
    humiture_indoor = dl.load(host=parser.con['host'], user=parser.con['user'], password=parser.con['password'],
                              port=parser.con['port'], database=parser.con['database'],
                              sql=parser.sql['humiture_indoor'])
    humiture_outdoor = dl.load(host=parser.con['host'], user=parser.con['user'], password=parser.con['password'],
                               port=parser.con['port'], database=parser.con['database'],
                               sql=parser.sql['humiture_outdoor'])
    elec_cooling_tower = dl.load(host=parser.con['host'], user=parser.con['user'], password=parser.con['password'],
                                 port=parser.con['port'], database=parser.con['database'],
                                 sql=parser.sql['cooling_tower'])
    data_obj = DataPrepare()
    y = data_obj.build_label(elec_cooling_tower, data_type='electricity')
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
