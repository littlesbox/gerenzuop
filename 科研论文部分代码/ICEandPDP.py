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

mpl.rcParams['figure.dpi'] = 300  # 设置全局 DPI

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'



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

# Add noise to the dataset
X_noisy = add_noise(X_scaled)

# Combine the original and noisy data
X_augmented = np.vstack((X_scaled, X_noisy))
y_augmented = np.hstack((y, y))

# 4. 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X_augmented, y_augmented, test_size=0.2, random_state=42)

mpl.rcParams['figure.dpi'] = 300  # 设置全局 DPI


#选择感兴趣的特征，数字为特征的索引
feature_to_plot = 0


#生成特征值范围
X_range = np.linspace(X_test[:, feature_to_plot].min(), X_test[:, feature_to_plot].max(), 100)
#设置布尔索引，用于复制样本的其他的特征值
idx = [True]*X_test.shape[1]
idx[feature_to_plot] = False
#创建曲线，其中ice_values中每个元素代表每个样本的y轴的取值
ice_values = []
#遍历所有样本，计算每个样本的y轴的值，即模型的预测值
for j in range(X_test.shape[0]):
    #创建当前样本的预测数据集，即感兴趣的特征的取值为之前设置的特征值范围，其他特征的取值为当前样本的特征值
    sample_data = np.zeros((len(X_range), X_test.shape[1]))
    for i in range(sample_data.shape[0]):
        sample_data[i,feature_to_plot] = X_range[i]
        sample_data[i,:][idx] = X_test[j,:][idx]
    #使用加载的模型进行预测
    y_pred = model.predict(sample_data)
    #保存当前样本的预测结果
    ice_values.append(y_pred)

#对感兴趣特征的取值进行逆标准化用于绘图展示，训练模型和进行预测的时候使用都是标准化之后的数据
X_range_plot = scaler.inverse_transform(sample_data)
X_range_plot = X_range_plot[:,feature_to_plot]



fig, ax = plt.subplots()

for item in ice_values:
    ax.plot(X_range_plot, item, alpha=0.1, color='#1f77b4') 

#将样本的预测值进行平均，用于绘制PDP图
#即，当固定一个感兴趣的特征的取值时，每个样本对应这一个预测值，计算这些样本的预测值的平均值
#ice_df的每一行对应着一个样本的曲线的y轴的取值，每一列对应着感兴趣的特征的取值
ice_df = pd.DataFrame(ice_values)
pdp = ice_df.mean()




ax.plot(X_range_plot, pdp, label="average",alpha=1, color='#ff7f0e')

plt.xlabel("Value range (mm)")  # X 轴标签
plt.ylabel("log[c(PAEs)]") 
plt.title(X.columns[feature_to_plot])
plt.legend()

plt.savefig(X.columns[feature_to_plot]+'.png',bbox_inches='tight')

plt.show()



# # 选择要绘制ICE图的特征索引（例如：MedInc, AveOccup）
# features = [22, 15, 1, 0, 30, 2]  # 0: MedInc, 5: AveOccup

# # 生成ICE曲线数据并绘制图像
# disp = PartialDependenceDisplay.from_estimator(
#     model, X_test, features, kind='both', 
#     ice_lines_kw={'color': 'blue', 'alpha': 0.2},
#     pd_line_kw={'color': 'red', 'linestyle': '--'}
# )
# plt.show()