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


mpl.rcParams['figure.dpi'] = 300  # 设置全局 DPI


# Set global font to Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'

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
ModelName = 'GBRT'
SaveDirectory = ModelName + '-' + formatted_datetime
if not os.path.exists(SaveDirectory):
    os.makedirs(SaveDirectory)

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




# # 1-1. 剔除异常值
# data['z_score'] = stats.zscore(data['PAEsConc'])
# del_indices = data['z_score'].abs() <= 3
# print(del_indices[del_indices == False].index)
# data = data[del_indices]
# data = data.drop(columns=['z_score'])


data = pd.concat([data0_cleaned,data1_cleaned])

data = data.drop(columns=['Z_Score'])
#数据变换
data['PAEsConc'] = np.log(data['PAEsConc'])
# data['PAEsConc'] = (data['PAEsConc'] ** 0.1 - 1) / 0.1

# 2. 选择特征和目标变量
X = data.iloc[:, 1:-1]  # 前31列为特征
y = data.iloc[:, -1]  # 最后一列为目标变量

# 3. 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Add noise to the dataset
X_noisy = add_noise(X_scaled, noiserdseed)

# Combine the original and noisy data
X_augmented = np.vstack((X_scaled, X_noisy))
y_augmented = np.hstack((y, y))

# 4. 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X_augmented, y_augmented, test_size=split_rio, random_state=split_rdseed)

# 5. 构建和训练模型
gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=model_cre_rdseed)
gbr.fit(X_train, y_train)

# 6. 模型评估
y_pred = gbr.predict(X_test)

# 7. 调优模型（可选）
param_grid = {
    # 'n_estimators': [200, 250, 300],  # 决策树的数量。更多的树可以提高模型的性能，但也会增加训练时间并可能导致过拟合。
    # 'learning_rate': [0.01, 0.05, 0.1],  # 每个树的贡献率。较低的学习率需要更多的树来达到相同的效果，较高的学习率则可能导致过拟合。
    # 'max_depth': [3, 4, 5],  # 每个决策树的最大深度。较深的树可以捕捉更多的细节，但也更容易过拟合。
    # 'min_samples_split': [2, 5, 10],  # 内部节点再划分所需的最小样本数。较大的值可以减少过拟合。
    # 'min_samples_leaf': [2, 4, 6, 8, 10],   # 叶节点所需的最小样本数。较大的值可以减少过拟合。
    # 'subsample': [0.6, 0.7, 0.8, 0.9, 1],   # 用于拟合每个决策树的样本比例。较小的值可以减少过拟合，但也可能增加方差
    # 'max_features': ['auto', 'sqrt', 'log2']   # 构建每个树时考虑的最大特征数。较小的值可以减少过拟合。
    # # Param
    'n_estimators': [100,150,200],  # 决策树的数量。更多的树可以提高模型的性能，但也会增加训练时间并可能导致过拟合。
    'learning_rate': [0.1,0.2],  # 每个树的贡献率。较低的学习率需要更多的树来达到相同的效果，较高的学习率则可能导致过拟合。
    'max_depth': [5,6],  # 每个决策树的最大深度。较深的树可以捕捉更多的细节，但也更容易过拟合。
    'min_samples_split': [3,4],  # 内部节点再划分所需的最小样本数。较大的值可以减少过拟合。
    'min_samples_leaf': [2,3],  # 叶节点所需的最小样本数。较大的值可以减少过拟合。
    'subsample': [0.5,0.6],  # 用于拟合每个决策树的样本比例。较小的值可以减少过拟合，但也可能增加方差
    'max_features': ['sqrt', 'log2']  # 构建每个树时考虑的最大特征数。较小的值可以减少过拟合。
    # PCA Param
    # 'n_estimators': [300],  # 决策树的数量。更多的树可以提高模型的性能，但也会增加训练时间并可能导致过拟合。
    # 'learning_rate': [0.05],  # 每个树的贡献率。较低的学习率需要更多的树来达到相同的效果，较高的学习率则可能导致过拟合。
    # 'max_depth': [4],  # 每个决策树的最大深度。较深的树可以捕捉更多的细节，但也更容易过拟合。
    # 'min_samples_split': [5],  # 内部节点再划分所需的最小样本数。较大的值可以减少过拟合。
    # 'min_samples_leaf': [2],  # 叶节点所需的最小样本数。较大的值可以减少过拟合。
    # 'subsample': [0.6],  # 用于拟合每个决策树的样本比例。较小的值可以减少过拟合，但也可能增加方差
    # 'max_features': [0.7]  # 构建每个树时考虑的最大特征数。较小的值可以减少过拟合。
}

grid_search = GridSearchCV(estimator=gbr, param_grid=param_grid, cv=cross_v, n_jobs=-1, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# 打印最佳参数
print(f"Best parameters found: {grid_search.best_params_}")

# 使用最佳参数训练模型
best_gbr = grid_search.best_estimator_
best_gbr.fit(X_train, y_train)

# 8. 使用优化后的模型进行预测（可选）
y_pred_optimized_scaled = best_gbr.predict(X_test)

# 计算额外的评估指标
mse = mean_squared_error(y_test, y_pred_optimized_scaled)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred_optimized_scaled)
r2 = r2_score(y_test, y_pred_optimized_scaled)

#保存评估指标
assess_list_value = [mse, rmse, mae, r2]
assess_list_name = ['MSE', 'RMSE', 'MAE', 'R2']
assess_list = pd.DataFrame({'Column1': assess_list_name, 'Column2': assess_list_value})
assess_list.to_csv(os.path.join(SaveDirectory, 'assess_list.csv'), index=False, header=False)

# 打印评估指标
print(f'MSE: {mse}')
print(f'RMSE: {rmse}')
print(f'MAE: {mae}')
print(f'R²: {r2}')

# 计算误差
errors = y_test - y_pred_optimized_scaled
# 绘制误差分布图
plt.figure(figsize=(8, 6))
# 绘制直方图
sns.histplot(errors, bins=20, color='skyblue', kde=False, stat='density')
# 绘制KDE曲线
sns.kdeplot(errors, color='red')
plt.xlabel('Prediction Error')
plt.ylabel('Density')
plt.title('Error Distribution Plot')
plt.savefig(os.path.join(SaveDirectory, 'Error Distribution Plot.png'),bbox_inches='tight')
plt.savefig(os.path.join(SaveDirectory, 'Error Distribution Plot.pdf'),bbox_inches='tight')
plt.show()
plt.close()

# 创建一个 TreeExplainer
explainer = shap.TreeExplainer(best_gbr)

# 计算SHAP值
shap_values = explainer.shap_values(X_train)

# 绘制特征重要性图
shap.summary_plot(shap_values[:, :-1], X_train[:, :-1], feature_names=X.columns[:-1], show=False)
plt.savefig(os.path.join(SaveDirectory, 'GBRT_shap_summary_plot.png'),bbox_inches='tight')
plt.savefig(os.path.join(SaveDirectory, 'GBRT_shap_summary_plot.pdf'),bbox_inches='tight')
plt.close()


# 提取特征重要性
importances = best_gbr.feature_importances_
feature_names = X.columns

# 将特征及其重要性分数进行排序
sorted_indices = importances.argsort()[::-1]
sorted_feature_names = [feature_names[i] for i in sorted_indices]
sorted_importances = importances[sorted_indices]
sum_imp = np.sum(sorted_importances)
per_imp = np.around((sorted_importances / sum_imp) * 100, 2)
perImpDF = pd.DataFrame(per_imp)
perImpDF.index = sorted_feature_names
perImpDF.to_excel(os.path.join(SaveDirectory, 'feature percentage_GBRT.xlsx'))

# # 可视化特征重要性
# plt.figure(figsize=(20, 15))
# plt.bar(sorted_feature_names[:10], sorted_importances[:10], color='skyblue')
# plt.xlabel('Feature')
# plt.ylabel('Importance')
# plt.title('Feature Importances')
# plt.xticks(rotation=45)
# plt.savefig(os.path.join(SaveDirectory, 'Feature Importances_GBRT.png'))
# plt.show()

# # 绘制学习曲线
# train_sizes, train_scores, test_scores = learning_curve(
#     best_gbr, X_train, y_train, cv=5, scoring='neg_mean_squared_error', n_jobs=-1,
#     train_sizes=np.linspace(0.1, 1.0, 10)
# )
#
# train_scores_mean = -train_scores.mean(axis=1)
# test_scores_mean = -test_scores.mean(axis=1)
#
# plt.figure(figsize=(8, 6))
# plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
# plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
# plt.title('Learning Curve')
# plt.xlabel('Training examples')
# plt.ylabel('Score')
# plt.legend(loc='best')
# plt.grid()
# plt.savefig(os.path.join(SaveDirectory, 'Learning Curve.png'))
# plt.show()

# # 实际值与预测值散点图
# # plt.figure()
# # plt.scatter(y_test, y_pred_optimized_scaled, alpha=0.5)
# # plt.xlabel('Actual values')
# # plt.ylabel('Predicted values')
# # plt.title('Actual vs Predicted values')
# # plt.grid()
# # plt.show()
#
# 实际值与预测值散点图
fig, ax = plt.subplots(figsize=(8, 6))
# 绘制散点图和拟合线（包括95%置信区间）
sns.regplot(x=y_test, y=y_pred_optimized_scaled,
            scatter_kws={'alpha': 0.5, 'color': 'green'},  # 散点颜色
            line_kws={'linewidth': 2},  # 拟合线颜色
            ci=95,
            ax=ax)

# 获取数据范围并稍微扩展一下，确保所有点都能显示
data_range = [
    min(min(y_test), min(y_pred_optimized_scaled)),
    max(max(y_test), max(y_pred_optimized_scaled))
]
margin = (data_range[1] - data_range[0]) * 0.05  # 添加5%的边距
plot_range = [data_range[0] - margin, data_range[1] + margin]

# 设置坐标轴范围
ax.set_xlim(plot_range)
ax.set_ylim(plot_range)

# 绘制对角线，使用与坐标轴完全相同的范围
x = np.array(plot_range)
ax.plot(x, x, color='r', linestyle='--')

# 计算线性回归参数
slope, intercept, r_value, p_value, std_err = stats.linregress(y_test, y_pred_optimized_scaled)

# 添加R2和MSE
text_str = f'R$^2$ = {round(r2, 4)}\nMSE = {round(mse, 4)}\nRMSE = {round(rmse, 4)}\nMAE = {round(mae, 4)}'
ax.text(0.02, 0.97, text_str, transform=ax.transAxes, fontsize=12, verticalalignment='top', horizontalalignment='left')
# 设置图形标题和标签
ax.set_title('Gradient Boosting Regression Tree')
ax.set_xlabel('Actual log[c(PAEs)]')
ax.set_ylabel('Predicted log[c(PAEs)]')
plt.savefig(os.path.join(SaveDirectory, 'GBRT.png'),bbox_inches='tight')
plt.savefig(os.path.join(SaveDirectory, 'GBRT.pdf'),bbox_inches='tight')
# # 添加图例
# ax.legend()
# 显示图形
plt.show()


# # 交叉验证结果箱线图
# cv_results = cross_val_score(best_gbr, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
# cv_results = -cv_results  # 将负的MSE转为正数
# plt.figure(figsize=(8, 6))
# plt.boxplot(cv_results, vert=False)
# plt.title('Cross-validation results')
# plt.xlabel('MSE')
# plt.grid()
# plt.savefig(os.path.join(SaveDirectory, 'Cross-validation results.png'))
# plt.show()


#保存重要参数
canshu = {
    'noiserdseed':[noiserdseed],
    'split_rio':[split_rio],
    'split_rdseed':[split_rdseed],
    'model_cre_rdseed':[model_cre_rdseed],
    'cross_v':[cross_v]
    }

canshu_df = pd.DataFrame(canshu)
canshu_df.to_csv(os.path.join(SaveDirectory, 'canshu_df.csv'), index=False)

param_grid_save = {
    'param':list(param_grid.keys()),
    'data':list(param_grid.values())
    }

param_grid_save_df = pd.DataFrame(param_grid_save)
param_grid_save_df.to_csv(os.path.join(SaveDirectory, 'param_grid_save_df.csv'), index=False)

best_params = {
    'param':list(grid_search.best_params_.keys()),
    'data':list(grid_search.best_params_.values())
    }

best_params_df = pd.DataFrame(best_params)
best_params_df.to_csv(os.path.join(SaveDirectory, 'best_params_df.csv'), index=False)



#保存训练好的模型
joblib.dump(best_gbr, os.path.join(SaveDirectory, 'best_gbr.joblib'))








# 获取 MSE 及其索引
mse_values = -grid_search.cv_results_['mean_test_score']  # 取负数，保证 MSE 是正数
indices = np.arange(1, len(mse_values) + 1)  # 自然数索引

# 绘制折线图
plt.figure(figsize=(10, 5))
plt.plot(indices, mse_values, linestyle='-', color='#18a6de', label='MSE')

# # 标注数据点
# for i, mse in zip(indices, mse_values):
#     plt.text(i, mse, f'{mse:.2f}', ha='right', va='bottom', fontsize=10)

plt.xlabel("Index")
plt.ylabel("MSE")
plt.title("MSE Changes")
#plt.xticks(indices)  # 设置横轴刻度
#plt.legend()
#plt.grid(True)
plt.savefig(os.path.join(SaveDirectory, 'cross_v.png'),bbox_inches='tight')
plt.show()
















