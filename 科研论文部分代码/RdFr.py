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
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import learning_curve, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score

import os
import joblib
from datetime import datetime

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
ModelName = 'RdFr'
SaveDirectory = ModelName + '-' + formatted_datetime
if not os.path.exists(SaveDirectory):
    os.makedirs(SaveDirectory)

# 1. 加载数据
file_path = 'traindata.xlsx'
data = pd.read_excel(file_path)
data['PAEsConc'] = np.log(data['PAEsConc'])

# 1-1. 剔除异常值
data['z_score'] = stats.zscore(data['PAEsConc'])
del_indices = data['z_score'].abs() <= 3
print(del_indices[del_indices == False].index)
data = data[del_indices]
data = data.drop(columns=['z_score'])


# 2. 选择特征和目标变量
X = data.iloc[:, :-1]  # 前31列为特征
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
rf = RandomForestRegressor(n_estimators=100, random_state=model_cre_rdseed)
rf.fit(X_train, y_train)

# 6. 模型评估
y_pred_scaled = rf.predict(X_test)

# 7. 调优模型（可选）
param_grid = {
    # 'n_estimators': [50, 100, 150, 200],  # 常见树的数量范围，50-200足够捕捉大多数数据集的特征 [(x+1)*5 for x in range(20)]
    # 'max_depth': [None, 5, 10, 15],  # 无限制和几个常见的深度限制，防止过拟合和欠拟合
    # 'min_samples_split': [2, 4, 6, 8],  # 节点分裂所需的最小样本数，默认是2，适当增大以减少过拟合
    # 'min_samples_leaf': [1, 2, 3, 4, 5],  # 叶节点所需的最小样本数，默认是1，适当增大以减少过拟合
    # 'max_features': ['auto', 'sqrt', 'log2'],  # 分裂时考虑的最大特征数，常见的选择是'auto'（所有特征）、'sqrt'（特征数的平方根）和'log2'（特征数的对数）
    # 'max_leaf_nodes': [None, 5, 10, 15, 20]  # 树中叶节点的最大数量，限制叶节点数以控制树的复杂度
    # Param
    'n_estimators': [62],  # 常见树的数量范围，50-200足够捕捉大多数数据集的特征 [(x+1)*5 for x in range(20)]
    'max_depth': [16],  # 无限制和几个常见的深度限制，防止过拟合和欠拟合
    'min_samples_split': [2],  # 节点分裂所需的最小样本数，默认是2，适当增大以减少过拟合
    'min_samples_leaf': [1],  # 叶节点所需的最小样本数，默认是1，适当增大以减少过拟合
    'max_features': [0.8],  # 分裂时考虑的最大特征数，常见的选择是'auto'（所有特征）、'sqrt'（特征数的平方根）和'log2'（特征数的对数）
    'max_leaf_nodes': [None]  # 树中叶节点的最大数量，限制叶节点数以控制树的复杂度
    # PCA Param
    # 'n_estimators': [90],  # 常见树的数量范围，50-200足够捕捉大多数数据集的特征 [(x+1)*5 for x in range(20)]
    # 'max_depth': [13],  # 无限制和几个常见的深度限制，防止过拟合和欠拟合
    # 'min_samples_split': [2],  # 节点分裂所需的最小样本数，默认是2，适当增大以减少过拟合
    # 'min_samples_leaf': [1],  # 叶节点所需的最小样本数，默认是1，适当增大以减少过拟合
    # 'max_features': [0.8],  # 分裂时考虑的最大特征数，常见的选择是'auto'（所有特征）、'sqrt'（特征数的平方根）和'log2'（特征数的对数）
    # 'max_leaf_nodes': [None]  # 树中叶节点的最大数量，限制叶节点数以控制树的复杂度
}

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=cross_v, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# 打印最佳参数
print(f"Best parameters found: {grid_search.best_params_}")

# 使用最佳参数训练模型
best_rf = grid_search.best_estimator_
best_rf.fit(X_train, y_train)

# 8. 使用优化后的模型进行预测（可选）
y_pred_optimized_scaled = best_rf.predict(X_test)

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

# # 计算误差
# errors = y_test - y_pred_optimized_scaled
# # 绘制误差分布图
# plt.figure(figsize=(8, 6))
# # 绘制直方图
# sns.histplot(errors, bins=20, color='skyblue', kde=False, stat='density')
# # 绘制KDE曲线
# sns.kdeplot(errors, color='red')
# plt.xlabel('Prediction Error')
# plt.ylabel('Density')
# plt.title('Error Distribution Plot')
# plt.savefig(os.path.join(SaveDirectory, 'Error Distribution Plot.png'))
# plt.savefig(os.path.join(SaveDirectory, 'Error Distribution Plot.pdf'))


# # 创建一个 TreeExplainer
# explainer = shap.TreeExplainer(best_rf)
#
# # 计算SHAP值
# shap_values = explainer.shap_values(X_test)
#
# # 绘制特征重要性图
# shap.summary_plot(shap_values, X_test, feature_names=X.columns, show=False)
# plt.savefig(os.path.join(SaveDirectory, 'RF_shap_summary_plot.png'))
# plt.savefig(os.path.join(SaveDirectory, 'RF_shap_summary_plot.pdf'))
# plt.close()
#
# # 绘制SHAP力图，选择第一个预测结果
# shap.force_plot(explainer.expected_value, shap_values[:, 0], X_test[:, 0], feature_names=X.columns, show=False)
# plt.savefig(os.path.join(SaveDirectory, 'RF_shap_force_plot.png'))
#
# # 绘制某个特定特征的SHAP依赖图，例如'feature_name'
# shap.dependence_plot(2, shap_values, X_test, feature_names=X.columns, show=False)
# plt.savefig(os.path.join(SaveDirectory, 'RF_shap_dependence_plot.png'))

# 提取特征重要性
importances = best_rf.feature_importances_
feature_names = X.columns

# 将特征及其重要性分数进行排序
sorted_indices = importances.argsort()[::-1]
sorted_feature_names = [feature_names[i] for i in sorted_indices]
sorted_importances = importances[sorted_indices]
sum_imp = np.sum(sorted_importances)
per_imp = np.around((sorted_importances / sum_imp) * 100, 2)
perImpDF = pd.DataFrame(per_imp)
perImpDF.index = sorted_feature_names
perImpDF.to_excel(os.path.join(SaveDirectory, 'feature percentage_RF.xlsx'))

#打印特征及其重要性分数
for feature, importance in zip(sorted_feature_names, sorted_importances):
    print(f"Feature: {feature}, Importance: {importance:.4f}")

# 可视化特征重要性
plt.figure(figsize=(20, 15))
plt.bar(sorted_feature_names[:10], sorted_importances[:10], color='skyblue')
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Feature Importances')
plt.xticks(rotation=45)
plt.savefig(os.path.join(SaveDirectory, 'Feature Importances_RF.png'))
plt.show()


# # 绘制学习曲线
# train_sizes, train_scores, test_scores = learning_curve(
#     best_rf, X_train, y_train, cv=5, scoring='neg_mean_squared_error', n_jobs=-1,
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
# plt.savefig(os.path.join(SaveDirectory, 'Learning Curve.pdf'))
# plt.show()
#
# 实际值与预测值散点图
fig, ax = plt.subplots(figsize=(8, 6))
# 绘制散点图
ax.scatter(y_test, y_pred_optimized_scaled, alpha=0.5)
x = np.linspace(y_test.min(), y_test.max(), 100)
# 添加直线 y = x
ax.plot(x, x, color='red')
# 添加R2和MSE
text_str = f'R² = {round(r2,4)}\nMSE = {round(mse,4)}'
ax.text(0.02, 0.97, text_str, transform=ax.transAxes, fontsize=12, verticalalignment='top', horizontalalignment='left')
# 设置图形标题和标签
ax.set_title('Random Forest Regression')
ax.set_xlabel('Actual log[c(PAEs)]')
ax.set_ylabel('Predicted log[c(PAEs)]')
plt.savefig(os.path.join(SaveDirectory, 'RandomForestRegression.pdf'))
# # 添加图例
# ax.legend()
# 显示图形
plt.show()
#
# # 交叉验证结果箱线图
# cv_results = cross_val_score(best_rf, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
# cv_results = -cv_results  # 将负的MSE转为正数
#
# plt.figure(figsize=(8, 6))
# plt.boxplot(cv_results, vert=False)
# plt.title('Cross-validation results')
# plt.xlabel('MSE')
# plt.grid()
# plt.savefig('Cross-validation results.pdf')
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
joblib.dump(best_rf, os.path.join(SaveDirectory, 'best_rf.joblib'))
