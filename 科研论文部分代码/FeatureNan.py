# -*- coding: utf-8 -*-
"""
特征值预测，根据时间序列来进行
注意特征值不可以为负值，因此要选择变换，对数变换时要注意 0 的情况
"""

import os
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.holtwinters import SimpleExpSmoothing



#读取文件夹下的所有文件
def list_files(directory):
    content1 = []
    content2 = []

    subdirectories = os.listdir(directory)
    
    for item in subdirectories:
        target = os.path.join(directory,item)
        
        #判断目标路径是不是文件夹
        if os.path.isdir(target):
            #如果是文件夹就进行递归
            content2 = content2 + list_files(target)
        else:
            content1.append(target)
     
    content = content1 + content2
    return content

#使用线性回归填充缺失值                              
def linear_interpolation(data):
    """使用线性回归填充缺失值"""
    df = pd.DataFrame({"time": np.arange(len(data)), "values": data})

    # 训练数据（去除缺失值）
    train_df = df.dropna()
    X_train = train_df["time"].values.reshape(-1, 1)
    y_train = train_df["values"].values

    # 训练线性回归模型
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 预测缺失值
    missing_indices = df[df["values"].isna()].index
    X_missing = df.loc[missing_indices, "time"].values.reshape(-1, 1)
    df.loc[missing_indices, "values"] = model.predict(X_missing)

    return df["values"].values

# 生成特征：使用过去 2*steplength 个时间点的值作为输入
def create_features(df, lag):
    for i in range(1, lag + 1):
        df[f"lag_{i}"] = df["values"].shift(i)
    return df

start_time = time.time()


#这是要预测的年份和月份######################################################
YearForPredict = 2027
MonthForPredict = 4
PredictTime = pd.to_datetime(str(YearForPredict)+'-'+str(MonthForPredict)+'-01')
PredictTime = PredictTime + pd.offsets.MonthEnd(0) #PredictTime为datetime格式
############################################################################

FeatureDir1 = '.\\FeatureForPredict\\NCDataFeature'
FeatureDir2 = '.\\FeatureForPredict\\YearBookFeature'

#获取每个需要预测的特征文件的路径，包括文件名
files_path = list_files(FeatureDir1) + list_files(FeatureDir2)
file_csv_path = []
#筛选出所有csv文件的路径，csv文件为特征文件
for item in files_path:
    houzhui = item.rsplit('.',1)
    if houzhui[1] == 'csv':
        file_csv_path.append(item)

#读取所有特征数据文件，文件名作为键，dataframe作为值，生成字典
FeatureDatalist = []
FeatureNamelist = []
for item in file_csv_path:
    df = pd.read_csv(item, sep=',', index_col = 0)
    dfname = os.path.basename(item)[:-4]
    FeatureDatalist.append(df)
    FeatureNamelist.append(dfname)
FeatureName2Data = dict(zip(FeatureNamelist, FeatureDatalist))

#读取特征信息文件
FeatureInfo = pd.read_csv('FeatureInfo.csv', sep=',', index_col = 0) 

param_grid_for_rf = {
    "n_estimators": [10,15,20,25],  # 树的数量
    "max_depth": [3,6,9,12],          # 树的最大深度
    "min_samples_split": [2,3,4]       # 最小样本分裂数
}

param_grid_for_gbrt = {
    "n_estimators": [50,75,100], # 迭代次数
    "learning_rate": [0.1,0.2,0.3], # 学习率
    "max_depth": [5,7,9] # 决策树最大深度
}


for item in list(FeatureName2Data.keys()):
    
    data = FeatureName2Data[item]
                
    nan_columns = data.columns[data.isnull().any()].tolist()
    
    if nan_columns != []:
        print(item)
        print(nan_columns)
        
        
        


end_time = time.time()  # 记录结束时间       
print(f"程序运行时间: {end_time - start_time:.6f} 秒")   