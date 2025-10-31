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
from scipy.interpolate import griddata

import os
import joblib
from datetime import datetime

from sklearn.inspection import permutation_importance
from sklearn.inspection import PartialDependenceDisplay

import matplotlib as mpl

from mpl_toolkits.mplot3d import Axes3D


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
# warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Function to add noise
def add_noise(data, noise_level=0.01):
    rng = np.random.default_rng(40)
    noise = rng.normal(0, noise_level, data.shape)
    return data + noise


#加载模型
model = joblib.load('Models\\xgb_model_best.joblib')



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

# 选择感兴趣的特征，数字为特征的索引
feature_to_plot = [19, 20]
# 生成特征值范围, 此时的范围还是标准化之后的范围，绘图的时候需要进行逆标准化
# x_range为第一个感兴趣特征的取值范围，y_range为第二个
x_range = np.linspace(X_test[:, feature_to_plot[0]].min()+0.2, X_test[:, feature_to_plot[0]].max(), 20)
y_range = np.linspace(X_test[:, feature_to_plot[1]].min()+0.6, X_test[:, feature_to_plot[1]].max(), 20)

# 将第一个特征的范围进行逆标准化,用于绘图
# 由于逆标准的方法要求的是二维表格，因此需要将当前特征值范围放入样本数据集中
idx = [True] * X_test.shape[1]
idx[feature_to_plot[0]] = False
sample_data = np.zeros((len(x_range), X_test.shape[1]))
for i in range(sample_data.shape[0]):
    sample_data[i, feature_to_plot[0]] = x_range[i]
    sample_data[i, :][idx] = X_test[0, :][idx]
x_range_plot = scaler.inverse_transform(sample_data)
x_range_plot = x_range_plot[:, feature_to_plot[0]]
# 将第二个特征的范围进行逆标准化，用于绘图
# 由于逆标准的方法要求的是二维表格，因此需要将当前特征值范围放入样本数据集中
idx = [True] * X_test.shape[1]
idx[feature_to_plot[1]] = False
sample_data = np.zeros((len(y_range), X_test.shape[1]))
for i in range(sample_data.shape[0]):
    sample_data[i, feature_to_plot[1]] = y_range[i]
    sample_data[i, :][idx] = X_test[0, :][idx]
y_range_plot = scaler.inverse_transform(sample_data)
y_range_plot = y_range_plot[:, feature_to_plot[1]]

# 利用两个特征的取值范围生成网格数据，用于绘制3D图像
# xax, yax为标准化之后的网格数据，xax_plot, yax_plot为标准化之前的网格数据
xax, yax = np.meshgrid(x_range, y_range)
xax_plot, yax_plot = np.meshgrid(x_range_plot, y_range_plot)
zvalues = np.zeros(xax.shape)

# 设置布尔索引，用于复制样本的其他的特征值
idx = [True] * X_test.shape[1]
idx[feature_to_plot[0]] = False
idx[feature_to_plot[1]] = False

# 计算每个样本在改变两个特征的值之后的预测结果
# 并且将这些结果求平均
for i in range(len(x_range)):
    for j in range(len(y_range)):
        sample_data = np.zeros(X_test.shape)
        sample_data[:, feature_to_plot] = [xax[i, j], yax[i, j]]
        sample_data[:, idx] = X_test[:, idx]
        z_pred = model.predict(sample_data)
        zvalues[i, j] = z_pred.mean()






fig = plt.figure(figsize=(6, 6))

ax1 = fig.add_subplot(1,1,1, projection='3d')

# 绘制曲面
surf = ax1.plot_surface(xax_plot, yax_plot, zvalues, cmap='YlGnBu')

# 去除网格
# ax.grid(False)

# 添加标题和标签
ax1.set_title(X.columns[feature_to_plot[0]]+' and '+X.columns[feature_to_plot[1]]+' Interact')
ax1.set_xlabel(X.columns[feature_to_plot[0]])
ax1.set_ylabel(X.columns[feature_to_plot[1]])
ax1.set_zlabel("log[c(PAEs)]")

# # 添加颜色条
# fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, location='left')

#plt.tight_layout()

plt.savefig(X.columns[feature_to_plot[0]]+'_'+X.columns[feature_to_plot[1]]+'_3D.png')
plt.savefig(X.columns[feature_to_plot[0]]+'_'+X.columns[feature_to_plot[1]]+'_3D.pdf')
# 显示图形
#plt.show()





x_new = np.linspace(xax_plot.min(), xax_plot.max(), 20)
y_new = np.linspace(yax_plot.min(), yax_plot.max(), 20)
x_grid, y_grid = np.meshgrid(x_new, y_new)
# 对原始数据进行重采样
points = np.column_stack((xax_plot.ravel(), yax_plot.ravel()))
z_grid = griddata(points, zvalues.ravel(), (x_grid, y_grid), method='linear')
# 创建新的图形
fig, ax = plt.subplots(figsize=(6, 5))
# 绘制低密度等高线图作为XY平面的投影，添加白色边缘
contour = ax.pcolormesh(x_grid, y_grid, z_grid, cmap='YlGnBu', edgecolors='grey', linewidth=0.25)
# 添加颜色条
cbar = fig.colorbar(contour)
cbar.set_label("log[c(PAEs)]")
# 设置轴标签
ax.set_xlabel(X.columns[feature_to_plot[0]])
ax.set_ylabel(X.columns[feature_to_plot[1]])
# # 设置标题
ax.set_title(X.columns[feature_to_plot[0]]+' and '+X.columns[feature_to_plot[1]]+' projection')


plt.tight_layout()

plt.savefig(X.columns[feature_to_plot[0]]+'_'+X.columns[feature_to_plot[1]]+'_projection.png', 
            bbox_inches='tight')
plt.savefig(X.columns[feature_to_plot[0]]+'_'+X.columns[feature_to_plot[1]]+'_projection.pdf', 
            bbox_inches='tight')
# 显示图形
plt.show()


