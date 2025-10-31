import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
import shap
import seaborn as sns
from scipy import stats
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve, cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

import os
import joblib
from datetime import datetime

from sklearn.inspection import permutation_importance
from sklearn.inspection import PartialDependenceDisplay

import matplotlib as mpl

from mpl_toolkits.mplot3d import Axes3D

# 忽略警告
#warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Function to add noise
def add_noise(data, noise_level=0.01):
    rng = np.random.default_rng(40)
    noise = rng.normal(0, noise_level, data.shape)
    return data + noise

#加载模型
model = joblib.load('Models\\xgb_model_best.joblib')

# #获取当前系统时间
# current_datetime = datetime.now()
# formatted_datetime = current_datetime.strftime("%Y-%m-%d-%H-%M-%S")
# #创建保存结果的文件夹，每次运行创建一个，文件夹的命名依据系统时间和模型名称
# ModelName = 'MLP'
# SaveDirectory = ModelName + '-' + formatted_datetime
# if not os.path.exists(SaveDirectory):
#     os.makedirs(SaveDirectory)


# 1. 加载数据
file_path = 'traindata.xlsx'
data = pd.read_excel(file_path)



#剔除异常值
data0 = data[data['PAEsType']==0].copy()
data1 = data[data['PAEsType']==1].copy()

data0['Z_Score'] = (data0['PAEsConc'] - data0['PAEsConc'].mean()) / data0['PAEsConc'].std()
data1['Z_Score'] = (data1['PAEsConc'] - data1['PAEsConc'].mean()) / data1['PAEsConc'].std()

threshold = 3

data0_cleaned = data0[abs(data0['Z_Score']) <= threshold]
data0_drop = data0[abs(data0['Z_Score']) > threshold]

data1_cleaned = data1[abs(data1['Z_Score']) <= threshold]
data1_drop = data1[abs(data1['Z_Score']) > threshold]


data = pd.concat([data0_cleaned,data1_cleaned])

data = data.drop(columns=['Z_Score'])
# # # 1-1. 剔除异常值
# data['z_score'] = stats.zscore(data['PAEs Conc.'])
# del_indices = data['z_score'].abs() <= 3
# print(del_indices[del_indices == False].index)
# data = data[del_indices]
# data = data.drop(columns=['z_score'])

# 2. 选择特征和目标变量
X = data.iloc[:, 1:-1]  # 前31列为特征
y = data.iloc[:, -1]  # 最后一列为目标变量

# 3. 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


#加载预测数据集
future_data_path = 'traindata_Future.xlsx'
future_data = pd.read_excel(future_data_path)
future_data1 = future_data.iloc[:,1:]


#标准化数据
future_data1_scaled = scaler.transform(future_data1)

#进行预测
future_data1_conc_log = model.predict(future_data1_scaled)

#将对数值转换成真实值
future_data1_conc = np.exp(future_data1_conc_log)

future_data['PAEsConc'] = future_data1_conc

#将PAEs含量数据加入到样本点数据集中

#保存模型预测的PAEs数据
# SaveDirectory = 'PAEs_Predicted_Data'
# if not os.path.exists(SaveDirectory):
#     os.makedirs(SaveDirectory)
# future_data.to_csv(os.path.join(SaveDirectory, 'fiveYrAft_PAEs.csv'), index=False)

future_data2 = future_data[['place','PAEsType','PAEsConc']]
#future_data2.to_csv('PAEsFutureConc.csv', index=False)


# data4 = data[['num','PAEsType','PAEsConc']]

# data4.to_csv('PAEsConcNOyichang.csv',index=False)




















