import numpy as np
import pandas as pd
from pe import permutation_entropy
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
# from sklearn_extra.cluster import KMedoids
from matplotlib import cm, pyplot as plt
import scipy.stats as st
import seaborn as sns


def sliding_window(X, sw_width, sw_steps):
    X = X.T
    start = 0
    row = X.shape[0]
    num = (row - sw_width) // sw_steps
    new_row = sw_width + (sw_steps * num)
    while True:
        if (start + sw_width) > new_row:
            return
        yield start, start + sw_width
        start += sw_steps

def data_process(data_dir="station01-1.xls", n=7):
    # 获取数据
    data = pd.read_excel(data_dir)
    cor = np.array((data[data["power"] > 0].corr("pearson")).iloc[-1, :7])
    # data['power'][data['power'] < 0] = 0
    data.date_time = pd.to_datetime(data.date_time)
    data.index = data['date_time']
    # 数据归一
    Data = data.drop('date_time', axis=1)
    scaler_for_x = MinMaxScaler(feature_range=(0, 1))
    Data = scaler_for_x.fit_transform(Data)
    Data = (pd.DataFrame(Data))
    Data.index = data.index
    Data.insert(loc=len(Data.columns), column='PV', value=data["power"])
    # 数据处理
    fea = {}
    for i in range(len(Data.columns) - 1):
        feature = (Data.values[:, i]).reshape(1, -1)
        fea[i] = []
        if i < 7:
            for start, end in sliding_window(feature, n, 1):
                fea[i].append((feature.T)[start:end, :])
            fea[i] = np.array(fea[i])
            fea[i] = fea[i].reshape(fea[i].shape[0], fea[i].shape[1])
            fea[i] = fea[i].astype(np.float)[1:, :]
        else:
            for start, end in sliding_window(feature, n+1, 1):
                fea[i].append((feature.T)[start:end, :])
            fea[i] = np.array(fea[i])
            fea[i] = fea[i].reshape(fea[i].shape[0], fea[i].shape[1])
            fea[i] = fea[i].astype(np.float)
    # 确定提前一至三步数据集
    nwp_fea_3 = np.hstack(
        (fea[0][:, 2:], fea[1][:, 2:], fea[2][:, 2:], fea[3][:, 2:], fea[4][:, 2:], fea[5][:, 2:], fea[6][:, 2:]))
    nwp_fea_2 = np.hstack((fea[0][:, 1:-1], fea[1][:, 1:-1], fea[2][:, 1:-1], fea[3][:, 1:-1], fea[4][:, 1:-1],
                           fea[5][:, 1:-1], fea[6][:, 1:-1]))
    nwp_fea_1 = np.hstack((fea[0][:, :-2], fea[1][:, :-2], fea[2][:, :-2], fea[3][:, :-2], fea[4][:, :-2],
                           fea[5][:, :-2], fea[6][:, :-2]))

    mea_fea = np.hstack((fea[7][:, :5], fea[8][:, :5], fea[9][:, :5], fea[10][:, :5], fea[11][:, :5], fea[12][:, :5]))

    X_fea_3 = np.hstack((nwp_fea_3, mea_fea, fea[13][:, :5], fea[13][:, -1].reshape(-1, 1)))
    X_fea_2 = np.hstack((nwp_fea_2, mea_fea, fea[13][:, :5], fea[13][:, -2].reshape(-1, 1)))
    X_fea_1 = np.hstack((nwp_fea_1, mea_fea, fea[13][:, :5], fea[13][:, -3].reshape(-1, 1)))
    # 负载序列
    load = (Data.values[:, -1]).reshape(1, -1)
    X = []
    for start, end in sliding_window(load, 12, 1):
        X.append((load.T)[start:end, :])
    X = np.array(X)
    X = X.reshape(X.shape[0], X.shape[1])
    X_train = X.astype(np.float)
    # 数据确认
    X_fea_1 = pd.DataFrame(X_fea_1)
    X_fea_1.insert(loc=len(X_fea_1.columns), column="PV",
                   value=Data.values[(Data.shape[0] - X_fea_1.shape[0]-2):-2, -1])
    
    # X_fea_1 = pd.DataFrame(X_fea_2)
    # X_fea_1.insert(loc=len(X_fea_1.columns), column="PV",
    #                value=Data.values[(Data.shape[0] - X_fea_1.shape[0]-1):-1, -1])
    
    # X_fea_1 = pd.DataFrame(X_fea_3)
    # X_fea_1.insert(loc=len(X_fea_1.columns), column="PV",
    #                value=Data.values[(Data.shape[0] - X_fea_1.shape[0]):, -1])
    
    X_fea_1[len(X_fea_1.columns) - 2] = X_fea_1["PV"]
    X_fea_1.index = data.index[7:]
    X_fea_1 = X_fea_1[X_fea_1["PV"] > 0]

    data_x = pd.DataFrame(X_train)
    data_x.index = data['date_time'][11:]
    data_x = data_x[data_x[len(data_x.columns) - 1] > 0]
    # 负载重构特征
    load = data_x
    load_mean = np.array(np.mean(load.iloc[:, :-1], axis=1)).reshape(-1, 1)
    load_std = np.array(np.std(load.iloc[:, :-1], axis=1)).reshape(-1, 1)
    load_pe = []

    for i in range(len(load)):
        load_pe.append(permutation_entropy(load.iloc[i, :-1]))

    # 功率序列不确定性
    load_pe = (np.array(load_pe)).reshape(-1, 1)
    load_diff = np.sum(np.diff(load.iloc[:, :-1]), axis=1).reshape(-1, 1)
    load_sum = np.hstack((load_mean, load_std, load_pe, load_diff))
    # NWP不确定性
    nwp = Data[Data["PV"] > 0].iloc[:, :7]
    nwp = cor * nwp

    # 聚类特征
    cluster_load = load.values
    feature = X_fea_1.values
    cluster_fea = np.hstack((load_sum, np.array(nwp)))

    return cluster_load, cluster_fea, feature, nwp

def get_data(k=4,n=7):
    cluster_load, cluster_fea, feature, nwp = data_process(data_dir="station01-1.xls", n=n)
    cluster_dic = {'data': {}, 'label': {}, 'feature': {}, 'cluster': {}, 'idx': {}, 'violin': {}}
    cluster_dic["data"] = cluster_load
    cluster_dic["label"] = cluster_dic["data"][:,-1]
    cluster_dic["feature"][0] = cluster_dic["data"][:,:-1]
    cluster_dic["cluster"][0] = cluster_fea

    km = KMeans(n_clusters=k, random_state=9)
    cluster_dic['idx'][0] = km.fit_predict(cluster_dic['cluster'][0])

    for i in range(k):
        cluster_dic['idx'][i + 1] = np.array(np.where(cluster_dic['idx'][0] == i))
    for i in range(k):
        cluster_dic['feature'][i + 1] = ((cluster_dic['feature'][0])[cluster_dic['idx'][i + 1], :]).\
            reshape(cluster_dic['idx'][i + 1].shape[1], 11)

    # 不同类别小提琴图对比
    plt.figure(figsize=(40, 25), dpi=300)
    font1 = {'family': 'Times New Roman',
             'size': 70, }

    cluster_dic['violin'] = pd.DataFrame((np.vstack((cluster_dic['label'], (cluster_dic['idx'][0] + 1)))).T)
    cluster_dic['violin']['class'] = ['class % i' % i for i in cluster_dic['violin'][1]]
    cluster_dic['violin'] = cluster_dic['violin'].sort_values(by=1, ascending=True, axis=0)


    sns.set(font_scale=5.5)
    sns.set_style("whitegrid")
    ax = sns.violinplot(x=0, y='class', data=cluster_dic['violin'])
    plt.ylabel('Class', font=font1)
    plt.xlabel('Corresponding power(KW)', font=font1)

    # 不同类别区间趋势图
    figure_dic = {}.fromkeys(['mean', 'std', 'low_CI_bound', 'high_CI_bound', 'pointplot', 'barplot'], None)  # 画图数据字典
    figure_dic['mean'] = {}.fromkeys([], [])  # 各类别均值
    figure_dic['std'] = {}.fromkeys([], [])  # 各类别标准差
    figure_dic['low_CI_bound'] = {}.fromkeys([], [])  # 区间下限
    figure_dic['high_CI_bound'] = {}.fromkeys([], [])  # 区间上限
    figure_dic['pointplot'] = {}.fromkeys([], [])
    figure_dic['barplot'] = {}.fromkeys([], [])

    for i in range(k):
        figure_dic['mean'][i] = pd.DataFrame(((np.sum(cluster_dic['feature'][i + 1], axis=0))
                                              / cluster_dic['feature'][i + 1].shape[0]).reshape(-1, 1))
        a = i + 1
        figure_dic['mean'][i].insert(1, 'class', ['class %a' % a] * 11)
        figure_dic['mean'][i].insert(2, 'position', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        figure_dic['std'][i] = np.std(cluster_dic['feature'][i + 1], axis=0)



    for i in range(k):
        figure_dic['low_CI_bound'][i], figure_dic['high_CI_bound'][i] = st.norm.interval(0.6,
                                                                                         loc=figure_dic['mean'][i][0],
                                                                                         scale=figure_dic['std'][i])
        figure_dic['low_CI_bound'][i] = figure_dic['low_CI_bound'][i].reshape(-1)
        figure_dic['high_CI_bound'][i] = figure_dic['high_CI_bound'][i].reshape(-1)
    figure_dic['mean'][k] = pd.concat([figure_dic['mean'][0], figure_dic['mean'][1],
                                       figure_dic['mean'][2], figure_dic['mean'][3]])

    plt.figure(figsize=(40, 20))
    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 70, }
    sns.set_context('talk', rc={"line.linewidth": 5})
    ax1 = sns.pointplot(x='position', y=0, data=figure_dic['mean'][k], hue='class',scale=2.5,
                        palette={"class 1": "g", "class 2": "m", "class 3": 'blue', 'class 4': 'r'},
                        markers=["^", "o", "D", "p"])
    ax2 = plt.fill_between(figure_dic['mean'][0].index, figure_dic['low_CI_bound'][0],
                           figure_dic['high_CI_bound'][0], alpha=0.6)
    ax2 = plt.fill_between(figure_dic['mean'][1].index, figure_dic['low_CI_bound'][1],
                           figure_dic['high_CI_bound'][1], alpha=0.6)
    ax2 = plt.fill_between(figure_dic['mean'][2].index, figure_dic['low_CI_bound'][2],
                           figure_dic['high_CI_bound'][2], alpha=0.6)
    ax2 = plt.fill_between(figure_dic['mean'][3].index, figure_dic['low_CI_bound'][3],
                           figure_dic['high_CI_bound'][3], alpha=0.6)

    plt.xticks(fontsize=60)
    plt.yticks(fontsize=60)
    plt.xlabel('Time stamp', fontdict=font1)
    plt.ylabel('Historical power(KW)', fontdict=font1)
    # plt.title('Confidence interval')
    ax1.legend(loc='center right', bbox_to_anchor=(0.85, 1), ncol=5)
    plt.setp(ax1.get_legend().get_texts(), fontsize='70')
    plt.show()

    # 数据整理
    data_dic = {'data': {}, 'class': {}, 'label': {}, 'position': {}, 'class_data': {}}

    data_dic['data'] = feature

    for i in range(k):
        data_dic['class'][i] = ((data_dic['data'])[cluster_dic['idx'][i + 1], :-2]).reshape(
            cluster_dic['idx'][i + 1].shape[1], 70)
        data_dic['label'][i] = ((data_dic['data'])[cluster_dic['idx'][i + 1], -1]).reshape(
            cluster_dic['idx'][i + 1].shape[1], 1)
        data_dic['position'][i] = (cluster_dic['idx'][i + 1]).T

    for i in range(k):
        data_dic['class_data'][i] = np.hstack((data_dic['class'][i],
                                               data_dic['label'][i], data_dic['position'][i]))

    return data_dic["class_data"]