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
from sklearn.preprocessing import MinMaxScaler
import os
import joblib
from datetime import datetime

from sklearn.inspection import permutation_importance

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

# 计算排列特征重要性
result = permutation_importance(model, X_test, y_test, n_repeats=100, random_state=42, scoring='neg_mean_squared_error')

feature_names = X.columns

# 显示特征重要性
importance_data = {
    'feature': feature_names,
    'importance': result.importances_mean,
}

importance = pd.DataFrame(importance_data)
importance['importance'] = np.abs(importance['importance'])
importance = importance.sort_values(by='importance', ascending=False)
print(importance)



importance['importance__scaled'] = importance['importance'] / importance['importance'].sum()
print(importance)
importance.to_excel('importance.xlsx')


# # 使用加载后的模型进行预测（可选）
# y_pred = best_mlp.predict(X_test)
#
# # 计算额外的评估指标
# mse = mean_squared_error(y_test, y_pred)
# rmse = np.sqrt(mse)
# mae = mean_absolute_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# #保存评估指标
# assess_list_value = [mse, rmse, mae, r2]
# assess_list_name = ['MSE', 'RMSE', 'MAE', 'R2']
# assess_list = pd.DataFrame({'Column1': assess_list_name, 'Column2': assess_list_value})
# assess_list.to_csv(os.path.join(SaveDirectory, 'assess_list.csv'), index=False, header=False)

# # 打印评估指标
# print(f'MSE: {mse}')
# print(f'RMSE: {rmse}')
# print(f'MAE: {mae}')
# print(f'R²: {r2}')


# # 实际值与预测值散点图
# fig, ax = plt.subplots(figsize=(8, 6))
# # 绘制散点图
# ax.scatter(y_test, y_pred, alpha=0.5)
# x = np.linspace(y_test.min(), y_test.max(), 100)
# # 添加直线 y = x
# ax.plot(x, x, color='red')
# # 添加R2和MSE
# text_str = f'R² = {round(r2,4)}\nMSE = {round(mse,4)}'
# ax.text(0.02, 0.97, text_str, transform=ax.transAxes, fontsize=12, verticalalignment='top', horizontalalignment='left')
# # 设置图形标题和标签
# ax.set_title('Multilayer Perceptron')
# ax.set_xlabel('Actual log[c(PAEs)]')
# ax.set_ylabel('Predicted log[c(PAEs)]')
# # plt.savefig(os.path.join(SaveDirectory, 'MLP.png'))
# # # 添加图例
# # ax.legend()
# # 显示图形
# plt.show()
