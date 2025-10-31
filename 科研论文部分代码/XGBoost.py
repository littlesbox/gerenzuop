'''
重要参数：
添加噪声数据的随机种子:noiserdseed
划分数据集时候的划分比例:split_rio,和随机种子:split_rdseed
交叉验证的倍数: cross_v
交叉验证寻优的param_grid，和寻找到的最优参数grid_search.best_params_
性能评估指标
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
import seaborn as sns
import shap
from scipy import stats
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing

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
cross_v = 3



#获取当前系统时间
current_datetime = datetime.now()
formatted_datetime = current_datetime.strftime("%Y-%m-%d-%H-%M-%S")
#创建保存结果的文件夹，每次运行创建一个，文件夹的命名依据系统时间和模型名称
ModelName = 'XGBoost'
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

# 将数据转换为DMatrix格式
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# 设置参数网格
param_grid = {
    # 'learning_rate': [0.01, 0.05, 0.1],  # 学习率，控制每个树的贡献
    # 'max_depth': [5, 6, 7],  # 树的最大深度，控制模型的复杂度
    # 'min_child_weight': [1, 3, 5],  # 子节点中最小的样本权重和，控制正则化程度
    # 'subsample': [0.3, 0.5, 0.7, 0.8, 0.9, 1.0],  # 用于训练模型的样本比例，控制过拟合
    # 'colsample_bytree': [0.4, 0.6, 0.8, 1.0],  # 用于训练每棵树的特征子样本比例，控制过拟合
    # 'n_estimators': [100, 200, 300]  # 树的数量，控制模型的复杂度
    # # Param
    'learning_rate': [0.05,0.1],  # 学习率，控制每个树的贡献
    'max_depth': [7,10],  # 树的最大深度，控制模型的复杂度
    'min_child_weight': [4, 5],  # 子节点中最小的样本权重和，控制正则化程度
    'subsample': [0.8,0.6],  # 用于训练模型的样本比例，控制过拟合
    'colsample_bytree': [0.65,0.7],  # 用于训练每棵树的特征子样本比例，控制过拟合
    'n_estimators': [200,300]  # 树的数量，控制模型的复杂度
    # PCA Param
    # 'learning_rate': [0.05],  # 学习率，控制每个树的贡献
    # 'max_depth': [6],  # 树的最大深度，控制模型的复杂度
    # 'min_child_weight': [4],  # 子节点中最小的样本权重和，控制正则化程度
    # 'subsample': [0.6],  # 用于训练模型的样本比例，控制过拟合
    # 'colsample_bytree': [0.7],  # 用于训练每棵树的特征子样本比例，控制过拟合
    # 'n_estimators': [300]  # 树的数量，控制模型的复杂度
}

# 转换为XGBoost参数格式
xgb_params = {
    'objective': 'reg:squarederror',
    'seed': 42
}

# # 使用GridSearchCV进行参数调优
# xgb_model = xgb.XGBRegressor(objective='reg:squarederror', seed=42)
# grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, n_jobs=-1,
#                            scoring='neg_mean_squared_error')
# grid_search.fit(X_train, y_train)
#
# # 打印最佳参数
# print(f"Best parameters found: {grid_search.best_params_}")
#
# # 使用最佳参数训练模型
# best_params = grid_search.best_params_
# xgb_params.update(best_params)
# num_round = best_params['n_estimators']
# xgb_params.pop('n_estimators', None)
#
# # 训练模型并记录评估结果
# evals_result = {}
# evals = [(dtrain, 'train'), (dtest, 'eval')]
# bst = xgb.train(xgb_params, dtrain, num_round, evals=evals, evals_result=evals_result, verbose_eval=True)
#
# # 预测
# y_pred = bst.predict(dtest)
#
# # 计算额外的评估指标
# mse = mean_squared_error(y_test, y_pred)
# rmse = np.sqrt(mse)
# mae = mean_absolute_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
#
# # 打印评估指标
# print(f'MSE: {mse}')
# print(f'RMSE: {rmse}')
# print(f'MAE: {mae}')
# print(f'R²: {r2}')


# XGBRegressor参数调优
xgb_model = xgb.XGBRegressor(**xgb_params)
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=cross_v, n_jobs=-1,
                           scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# 打印最佳参数
print(f"Best parameters found: {grid_search.best_params_}")

# 使用最佳参数重新初始化模型
best_params = grid_search.best_params_
xgb_model_best = xgb.XGBRegressor(**best_params, objective='reg:squarederror', seed=42)
# 训练模型
xgb_model_best.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=True)
# 预测
y_pred = xgb_model_best.predict(X_test)
# 计算均方误差
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)


print(f'MSE: {mse}')
print(f'RMSE: {rmse}')
print(f'MAE: {mae}')
print(f'R²: {r2}')

#保存评估指标
assess_list_value = [mse, r2,rmse,mae]
assess_list_name = ['MSE', 'R2','RMSE','MAE']
assess_list = pd.DataFrame({'Column1': assess_list_name, 'Column2': assess_list_value})
assess_list.to_csv(os.path.join(SaveDirectory, 'assess_list.csv'), index=False, header=False)

# 计算误差
errors = y_test - y_pred
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

# # 创建一个TreeExplainer对象
# explainer = shap.TreeExplainer(xgb_model_best)
#
# # 计算SHAP值
# shap_values = explainer.shap_values(X_train)
#
# # 绘制总结图并保存为PNG文件
# shap.summary_plot(shap_values, X_train, feature_names=X.columns, show=False)
# plt.savefig('shap_summary_plot_PCA.png')
# plt.savefig('shap_summary_plot_PCA.pdf')
# plt.close()

# # 绘制依赖图并保存为PNG文件
# shap.dependence_plot(0, shap_values, X_train, feature_names=X.columns, show=False)
# plt.savefig('shap_dependence_plot.png')
# plt.close()
#
# # 强制图（Force Plot）
# shap.initjs()
# force_plot = shap.force_plot(explainer.expected_value, shap_values[0], X_train[0], feature_names=X.columns)
#
# # 保存为HTML文件
# shap.save_html('shap_force_plot.html', force_plot)
#
# # 绘制值条形图并保存为PNG文件
# shap.summary_plot(shap_values, X_train, plot_type="bar", feature_names=X.columns, show=False)
# plt.savefig('shap_bar_plot.png')
# plt.close()


# # 绘制第0棵树的结构
# plt.figure(figsize=(100, 80))
# dot_data = xgb.to_graphviz(bst, num_trees=0, rankdir='LR')  # rankdir='LR'表示从左到右显示
# dot_data.render('xgboost_tree', format='pdf')




# # 提取特征重要性
# importance = bst.get_score(importance_type='weight')
# feature_names = X.columns
#
# # 将特征及其重要性分数进行排序
# importance_df = pd.DataFrame({
#     'Feature': feature_names,
#     'Importance': importance.values()
# }).sort_values(by='Importance', ascending=False)
# sum_imp = importance_df['Importance'].sum()
# importance_df['Importance'] = (importance_df['Importance'] / sum_imp * 100).round(2)
# importance_df.to_excel(os.path.join(SaveDirectory, 'feature percentage_XGBoost.xlsx', index=False))
#
# # 可视化特征重要性
# plt.figure(figsize=(20, 15))
# plt.bar(importance_df['Feature'][:10], importance_df['Importance'][:10], color='skyblue')
# plt.xlabel('Feature')
# plt.ylabel('Importance')
# plt.title('Feature Importance')
# plt.xticks(rotation=45)
# plt.savefig(os.path.join(SaveDirectory, 'Feature Importance_XGBoost.png'))
# plt.show()

# # 提取训练和验证的评估结果
# train_rmse = evals_result['train']['rmse']
# eval_rmse = evals_result['eval']['rmse']
#
# # 绘制学习曲线
# epochs = len(train_rmse)
# x_axis = range(0, epochs)
# plt.figure(figsize=(8, 6))
# plt.plot(x_axis, train_rmse, label='Training score', color='r')
# plt.plot(x_axis, eval_rmse, label='Cross-validation score', color='g')
# plt.fill_between(x_axis, train_rmse, alpha=0.1, color='r')
# plt.fill_between(x_axis, eval_rmse, alpha=0.1, color='g')
# plt.title('Learning Curve')
# plt.xlabel('Boosting Rounds')
# plt.ylabel('RMSE')
# plt.legend(loc='best')
# plt.grid(True)
# plt.savefig('Learning Curve.pdf')
# plt.show()
#
#
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
# ax.set_title('Extreme Gradient Boosting')
# ax.set_xlabel('Actual log[c(PAEs)]')
# ax.set_ylabel('Predicted log[c(PAEs)]')
# plt.savefig(os.path.join(SaveDirectory, 'XGBoost.png'))
# # # 添加图例
# # ax.legend()
# # 显示图形
# plt.show()
#
#
# # 进行交叉验证并获取结果
# cv_results = xgb.cv(xgb_params, dtrain, num_boost_round=100, nfold=5, metrics={'rmse'}, seed=42, shuffle=True,
#                     as_pandas=True)
# # 提取每个折的结果
# mse_scores = cv_results['test-rmse-mean'] ** 2  # 转换为MSE
# # 创建一个DataFrame来存储交叉验证结果
# cv_results_df = pd.DataFrame({'MSE': mse_scores})
# # 绘制箱型图
# plt.figure(figsize=(8, 6))
# box_plot = sns.boxplot(data=cv_results_df, orient='h', width=0.14, boxprops=dict(facecolor='None'))
# plt.title('Cross-validation results')
# plt.xlabel('MSE')
# plt.grid(True)
# plt.savefig('Cross-validation results.pdf')
# plt.show()

# 实际值与预测值散点图
fig, ax = plt.subplots(figsize=(8, 6))
# 绘制散点图和拟合线（包括95%置信区间）
sns.regplot(x=y_test, y=y_pred,
            scatter_kws={'alpha': 0.5, 'color': 'green'},  # 散点颜色
            line_kws={'linewidth': 2},  # 拟合线颜色
            ci=95,
            ax=ax)

# 获取数据范围并稍微扩展一下，确保所有点都能显示
data_range = [
    min(min(y_test), min(y_pred)),
    max(max(y_test), max(y_pred))
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
slope, intercept, r_value, p_value, std_err = stats.linregress(y_test, y_pred)

# 添加R2和MSE
text_str = f'R$^2$ = {round(r2, 4)}\nMSE = {round(mse, 4)}\nRMSE = {round(rmse, 4)}\nMAE = {round(mae, 4)}'
ax.text(0.02, 0.97, text_str, transform=ax.transAxes, fontsize=12, verticalalignment='top', horizontalalignment='left')
# 设置图形标题和标签
ax.set_title('Extreme Gradient Boosting')
ax.set_xlabel('Actual log[c(PAEs)]')
ax.set_ylabel('Predicted log[c(PAEs)]')
plt.savefig(os.path.join(SaveDirectory, 'Extreme Gradient Boosting.png'),bbox_inches='tight')
plt.savefig(os.path.join(SaveDirectory, 'Extreme Gradient Boosting.pdf'),bbox_inches='tight')
# # 添加图例
# ax.legend()
# 显示图形
plt.show()








#保存重要参数
canshu = {
    'noiserdseed':[noiserdseed],
    'split_rio':[split_rio],
    'split_rdseed':[split_rdseed],
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
joblib.dump(xgb_model_best, os.path.join(SaveDirectory, 'xgb_model_best.joblib'))