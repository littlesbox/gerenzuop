'''
重要参数：
添加噪声数据的随机种子:noiserdseed
划分数据集时候的划分比例:split_rio,和随机种子:split_rdseed
构建模型时的随机种子：model_cre_rdseed
交叉验证的倍数: cross_v
交叉验证寻优的param_grid，和寻找到的最优参数grid_search.best_params_
性能评估指标
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import seaborn as sns
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import learning_curve, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib as mpl
import os
import joblib
from datetime import datetime


mpl.rcParams['figure.dpi'] = 600  # 设置全局 DPI


# Function to add noise
def add_noise(data, noiserdseed, noise_level=0.01):
    rng = np.random.default_rng(noiserdseed)
    noise = rng.normal(0, noise_level, data.shape)
    return data + noise


noiserdseed = 40
split_rio = 0.2
split_rdseed = 42
model_cre_rdseed = 42
cross_v = 3


#获取当前系统时间
current_datetime = datetime.now()
formatted_datetime = current_datetime.strftime("%Y-%m-%d-%H-%M-%S")
#创建保存结果的文件夹，每次运行创建一个，文件夹的命名依据系统时间和模型名称
# ModelName = 'GBRT'
# SaveDirectory = ModelName + '-' + formatted_datetime
# if not os.path.exists(SaveDirectory):
#     os.makedirs(SaveDirectory)

# 1. 加载数据
file_path = 'traindata.csv'
data = pd.read_csv(file_path, sep=',')





#可视化展示最后一列
plt.figure()
dnbpvalue = data[data['PAEsType']==0]['PAEsConc']
dnbpnum = data[data['PAEsType']==0]['num']
#plt.scatter(dnbpnum, dnbpvalue, color='b', marker='o')
#plt.grid(True)
sns.scatterplot(x=dnbpnum, y=dnbpvalue)
# 设置标签和标题
plt.xlabel("Sample num")
plt.ylabel("DnBP Conc")
#plt.title("")
#plt.savefig(".\\Figure\\DnBP Conc.png", dpi=300, bbox_inches='tight')

plt.figure()
dehpvalue = data[data['PAEsType']==1]['PAEsConc']
dehpnum = data[data['PAEsType']==1]['num']
#plt.scatter(dnbpnum, dnbpvalue, color='b', marker='o')
#plt.grid(True)
sns.scatterplot(x=dehpnum, y=dehpvalue)
# 设置标签和标题
plt.xlabel("Sample num")
plt.ylabel("DEHP Conc")
#plt.title("")
#plt.savefig(".\\Figure\\DEHP Conc.png", dpi=300, bbox_inches='tight')




data0 = data[data['PAEsType']==0].copy()
data1 = data[data['PAEsType']==1].copy()

data0['Z_Score'] = (data0['PAEsConc'] - data0['PAEsConc'].mean()) / data0['PAEsConc'].std()
data1['Z_Score'] = (data1['PAEsConc'] - data1['PAEsConc'].mean()) / data1['PAEsConc'].std()

threshold = 3

data0_cleaned = data0[abs(data0['Z_Score']) <= threshold]
data0_drop = data0[abs(data0['Z_Score']) > threshold]

data1_cleaned = data1[abs(data1['Z_Score']) <= threshold]
data1_drop = data1[abs(data1['Z_Score']) > threshold]



#可视化展示最后一列
plt.figure()
#plt.scatter(dnbpnum, dnbpvalue, color='b', marker='o')
#plt.grid(True)
sns.scatterplot(x=data0_cleaned['num'], y=data0_cleaned['PAEsConc'])
# 设置标签和标题
plt.xlabel("Sample num")
plt.ylabel("DnBP Conc")
#plt.title("")
#plt.savefig(".\\Figure\\clean DnBP Conc.png", dpi=600, bbox_inches='tight')

plt.figure()
#plt.scatter(dnbpnum, dnbpvalue, color='b', marker='o')
#plt.grid(True)
sns.scatterplot(x=data1_cleaned['num'], y=data1_cleaned['PAEsConc'])
# 设置标签和标题
plt.xlabel("Sample num")
plt.ylabel("DEHP Conc")
#plt.title("")
#plt.savefig(".\\Figure\\clean DEHP Conc.png", dpi=600, bbox_inches='tight')


data = data0_cleaned.drop(columns=['Z_Score','PAEsType'])

#数据变换
data['PAEsConc'] = np.log(data['PAEsConc'])

# 2. 选择特征和目标变量
X = data.iloc[:, 1:-1]  # 第2列到倒数第二列为特征
y = data.iloc[:, -1]  # 最后一列为目标变量

# 3. 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_scaled_df = pd.DataFrame(X_scaled)
X_scaled_df.columns = X.columns
corr_matrix = X_scaled_df.corr()
    
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, square=True)
plt.savefig(".\\Figure\\corr_matrix.png", dpi=600, bbox_inches='tight')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    