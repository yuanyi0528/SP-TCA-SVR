import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal
import seaborn as sns
from TCA import TCA
import itertools
from dataloader import get_data
from sklearn import svm
from sklearn.ensemble import GradientBoostingRegressor
from elm import ELMRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV

def domain_adaption(data,n=65):

    tca = TCA(dim=n)

    for i in range(len(data)):
        data_nwp = data[i][:,:n]
        data_pv = data[i][:,n:]
        m = int(data_nwp.shape[0]*0.7)
        Ps = data_nwp[:m,:]
        Pt = data_nwp[m:,:]
        Ps_new, Pt_new = tca.fit(Ps, Pt)
        data[i] = np.hstack((np.vstack((Ps_new,Pt_new)), data_pv))

    return data

def model_regression(X):
    n = int(X.shape[0] * 0.7)
    Xs = X[:n, :-2]
    Ys = (X[:n, -2]).reshape(-1, 1)
    Xt = X[n:, :-2]
    Yt = (X[n:, -2]).reshape(-1, 1)
    position = (X[n:, -1]).reshape(-1, 1)
    # model_1 = GradientBoostingRegressor()
    model_1 = svm.SVR()
    # model_1 = ELMRegressor(n_hidden=60, alpha=0.9, rbf_width=1, activation_func="sigmoid")
    param = [{'kernel':['linear', 'rbf', 'sigmoid'],'degree':[3, 4, 5, 6],
              'tol':[0.005, 0.008, 0.01, 0.013, 0.015, 0.017, 0.02]}]

    grid = GridSearchCV(model_1, param_grid=param, cv=10, verbose=0)
    # grid = model_1
    grid.fit(Xs, Ys)
    print('最优分类器:',grid.best_params_,'最优分数:', grid.best_score_)
    Y_pre = (grid.predict(Xt)).reshape(-1, 1)
    Y = np.hstack((Y_pre, Yt, position))
    return Y


Data_dic = {"data":{}, "pre":{}}
Data_dic['data'] = pd.read_excel("station01-1.xls")
Data_dic['data'].date_time=pd.to_datetime(Data_dic['data'].date_time)
Data_dic['data'].index = Data_dic['data']['date_time']
Data_dic['data'] = Data_dic['data'].drop('date_time',axis=1)
Data_dic['data'] = Data_dic['data'][Data_dic['data']['power'] > 0]

data = get_data(k=4,n=7)
data_class = domain_adaption(data)
for i in range(len(data_class)):
    Data_dic["pre"][i] = model_regression(data_class[i])
pre = np.vstack((Data_dic["pre"][0], Data_dic["pre"][1], Data_dic["pre"][2],
                 Data_dic["pre"][3]))

pre = pre[np.lexsort(pre.T)]

R2 = r2_score(pre[:,1],pre[:,0])
mse = mean_squared_error(pre[:,1],pre[:,0])
mae = mean_absolute_error(pre[:,1],pre[:,0])








