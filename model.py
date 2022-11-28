# encoding: utf-8
"""
#@file: model.py
#@time: 2022-08-23 14:45
#@author: ywtai 
#@contact: 632910913@qq.com
#@software: PyCharm
#@desc:
"""
import os
import xgboost as xgb
from hyperopt import hp, fmin, tpe, STATUS_OK, STATUS_FAIL, Trials
from hyperopt.early_stop import no_progress_loss
from sklearn.model_selection import cross_val_score
from functools import partial
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from scipy.stats import norm
import joblib


class HPOpt(object):

    def __init__(self, x_train, y_train, x_test=None, y_test=None):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def run(self, fn_name, space, trials, algo, max_evals, early_stopping=10):
        fn = getattr(self, fn_name)
        try:
            result = fmin(fn=fn, space=space, algo=algo, max_evals=max_evals, trials=trials,
                          early_stop_fn=no_progress_loss(early_stopping), return_argmin=False)
        except Exception as e:
            return {'status': STATUS_FAIL,
                    'exception': str(e)}
        return result, trials

    def xgb_reg(self, para):
        reg = xgb.XGBRegressor(**para['reg_params'])
        return self.train_reg(reg, para)

    def train_reg(self, reg, para):
        reg.fit(self.x_train, self.y_train, **para['fit_params'])
        cv_loss = -cross_val_score(estimator=reg, X=self.x_train, y=self.y_train, cv=10,
                                   scoring='neg_root_mean_squared_error')
        # pred = reg.predict(self.x_test)
        # loss = np.sqrt(mean_squared_error(self.y_test, pred)) + cv_loss.mean() + 2 * np.abs(np.mean(self.y_test) - np.mean(pred))
        pred = reg.predict(self.x_train)
        loss = cv_loss.mean() + np.abs(np.mean(self.y_train) - np.mean(pred))
        return {'loss': loss, 'status': STATUS_OK}


class XgbModel:

    def __init__(self):
        xgb_reg_params = {
            'learning_rate': hp.quniform('learning_rate', 0.01, 1, 0.01),
            'max_depth': hp.choice('max_depth', np.arange(3, 13, 1)),
            'min_child_weight': hp.choice('min_child_weight', np.arange(1, 11, 1)),
            'colsample_bytree': hp.quniform('colsample_bytree', 0.1, 1, 0.1),
            'subsample': hp.quniform('subsample', 0.1, 1, 0.01),
            'n_estimators': hp.choice('n_estimators', np.arange(5, 100, 1)),
            'gamma': hp.quniform('gamma', 0, 0.5, 0.01),
            'reg_lambda': hp.quniform('reg_lambda', 0, 10, 0.1),
            'reg_alpha': hp.quniform('reg_alpha', 0, 1, 0.01),
        }
        xgb_fit_params = {
            'verbose': False
        }
        self.xgb_para = dict()
        self.xgb_para['reg_params'] = xgb_reg_params
        self.xgb_para['fit_params'] = xgb_fit_params
        self.best_estimator_ = None

    def fit(self, x, y):
        # 为了便于xgboost进行超参数搜索，对label进行伪归一化
        base_divisor = np.percentile(y, 0.75)
        use_y = y / base_divisor
        obj = HPOpt(x, use_y)
        algo = partial(tpe.suggest, n_startup_jobs=125, n_EI_candidates=50)
        xgb_opt = obj.run(fn_name='xgb_reg', space=self.xgb_para, trials=Trials(), algo=algo, max_evals=1000,
                          early_stopping=100)
        print('search opt parameter:')
        print(xgb_opt[0]['reg_params'])
        reg = xgb.XGBRegressor(**xgb_opt[0]['reg_params'])
        reg.fit(x, use_y)
        self.best_estimator_ = reg
        self.best_estimator_.base_divisor = base_divisor
        return self

    def predict(self, x):
        return np.abs(self.best_estimator_.predict(x) * self.best_estimator_.base_divisor)

    def save(self, save_path=None):
        if save_path is None:
            save_path = os.path.join('model', 'xgb_model.json')
        parent_path = os.path.dirname(save_path)
        if not os.path.exists(parent_path):
            os.makedirs(parent_path)
        if self.best_estimator_ is not None:
            self.best_estimator_.save_model(save_path)

    def load(self, load_path=None):
        if load_path is None:
            load_path = os.path.join('model', 'xgb_model.json')
        if os.path.exists(load_path):
            if self.best_estimator_ is None:
                self.best_estimator_ = xgb.XGBRegressor()
            self.best_estimator_.load_model(load_path)

    def get_model(self):
        return self.best_estimator_


class KDMSample:

    def __init__(self, need_iter=3000):
        self.need_iter = need_iter
        self.best_estimator_ = None
        self.kde_sample = None

    def fit(self, x):
        kde = KernelDensity(kernel='gaussian')
        model_input = x.reshape(-1, 1)
        parameters = {'bandwidth': np.linspace(0.001, 0.5, 1000)}
        clf = GridSearchCV(kde, parameters)
        clf.fit(model_input)
        best_model = clf.best_estimator_
        print('best model:')
        print(best_model)
        best_model.fit(model_input)
        # 得到概率密度函数
        kde_sample = best_model.score_samples
        self.best_estimator_ = best_model
        self.kde_sample = kde_sample
        return self

    def sample(self, need_samples=10):
        """
        根据x通过核密度估计得到x的概率密度，在通过M-H采样基于该密度进行采样
        """
        current_x = 0
        sub_sample = []
        for i in range(self.need_iter * 2):
            if len(sub_sample) > need_samples:
                break
            next_x = norm.rvs(loc=current_x, scale=1, size=1)[0]
            alpha = min(1, np.exp(self.kde_sample([[next_x]]))[0] / np.exp(self.kde_sample([[current_x]]))[0])
            u = np.random.uniform(0, 1)
            if u < alpha and next_x > 0:
                current_x = next_x
                if i >= self.need_iter:
                    sub_sample.append(current_x)
        return np.array(sub_sample)

    def save(self, save_path=None):
        if save_path is None:
            save_path = os.path.join('model', 'kdm_sample.model')
        parent_path = os.path.dirname(save_path)
        if not os.path.exists(parent_path):
            os.makedirs(parent_path)
        if self.best_estimator_ is not None:
            joblib.dump(self.best_estimator_, save_path)

    def load(self, load_path=None):
        if load_path is None:
            load_path = os.path.join('model', 'kdm_sample.model')
        if os.path.exists(load_path):
            self.best_estimator_ = joblib.load(load_path)
            self.kde_sample = self.best_estimator_.score_samples

    def get_model(self):
        return self.best_estimator_
