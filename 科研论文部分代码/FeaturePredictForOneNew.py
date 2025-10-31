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
from statsmodels.stats.diagnostic import acorr_ljungbox
import seaborn as sns
import statsmodels.api as sm



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



item = 'P_fertilizer'



data = FeatureName2Data[item]

nan_cols = data.columns[data.isna().all()].tolist()
print(item,nan_cols)

#获取当前特征信息
FeatureName = item #这里的FeatureName不带后缀名
FeatureTimeResolution = FeatureInfo.loc[FeatureName, 'TimeResolution']

#根据当前的特征的时间分辨率，确定要预测的步长，并且生成新的索引（要用与构建特征）
if FeatureTimeResolution == 'Y':
    DataEndYear = int(data.index.tolist()[-1][0:4])
    steplength = YearForPredict - DataEndYear
    FutureIndex = [DataEndYear+1+x for x in range(steplength)]
    FutureIndex = [str(x)+'-12-31' for x in FutureIndex]
elif FeatureTimeResolution == 'M':
    DataEndTime = data.index.tolist()[-1]
    DataEndTime = pd.to_datetime(DataEndTime)
    steplength = (PredictTime.year - DataEndTime.year) * 12 + (PredictTime.month - DataEndTime.month)
    next_month_date = DataEndTime + pd.DateOffset(months=1)
    FutureIndex = pd.date_range(start=next_month_date, periods=steplength, freq="ME")
    FutureIndex = FutureIndex.strftime("%Y-%m-%d").tolist()
    
columnsName = data.columns.tolist()
FutureData = {}


element = columnsName[3]

sequenceOrigin = data[element].tolist()     
#对sequenceOrigin进行变换，nan可以参与下面的运算，结果仍是nan, 西藏的Particulates有nan
sequenceOrigin1 = [x+1e-6 for x in sequenceOrigin]
sequenceOriginlog = list(np.log(sequenceOrigin1))

# 判断列表中是否有nan，填充缺失值, 用线性回归
has_nan = pd.isna(sequenceOriginlog).any()
if has_nan:
    sequenceOriginlog = linear_interpolation(sequenceOriginlog)

#对sequenceOriginlog进行标准化
mean_value = np.mean(sequenceOriginlog)
std_value = np.std(sequenceOriginlog)
sequence = [(x-mean_value)/std_value for x in sequenceOriginlog]
print(max(sequence))
print(min(sequence))      
Train_sequence = sequence[0:len(sequence)-steplength] #训练序列
Test_sequence = sequence[len(sequence)-steplength:] #测试序列

mse_dict = {}
model_dict = {}
    
##########################################################################################
'''
序列创建完成，可以进行时间序列预测
重要参数说明：
steplength 是预测的时间步，是一个整数
FutureIndex 是每一个预测时间步所对应的索引，是一个列表，里面是日期字符串
            索引用于将预测的值存入dataframe中， 保持数据格式一致性，后面
            将用于构建特征
sequence 是用来进行时间预测的序列
Train_sequence 用于训练模型，等于 sequence - Test_sequence
Test_sequence 用于测试模型，长度等于 steplength

'''
##########################################################################################


##########################################################################################
'''ARIMA'''###############################################################################
##########################################################################################
#创建时间索引 
df = pd.DataFrame({"values": Train_sequence})

# **自动选择 ARIMA (p, d, q) 参数**
model_auto = auto_arima(df["values"], 
                        seasonal=False,  # 非季节性数据
                        trace=True,      # 显示优化过程
                        error_action="ignore", 
                        suppress_warnings=True, 
                        stepwise=True)   # 逐步搜索最佳参数

print(f"\n最佳 ARIMA 参数: {model_auto.order}")  # 输出自动选择的 (p, d, q)

# **训练 ARIMA 模型**
model = ARIMA(df["values"], order=model_auto.order)
ARIMA_model = model.fit()

# **滚动预测未来 steplength 个时间步**
n_steps = steplength  # 预测未来 steplength 个时间点
future_preds = ARIMA_model.forecast(steps=n_steps)

#计算在测试集上的mse
ARIMA_mse = np.mean((np.array(Test_sequence) - np.array(future_preds)) ** 2)
#保存模型误差
mse_dict['ARIMA'] = ARIMA_mse
# 保存模型
model_dict['ARIMA'] = ARIMA_model


##########################################################################################
'''SARIMA'''##############################################################################
##########################################################################################
#只有在特征的时间分辨率为 月 的时候使用
if FeatureTimeResolution == 'M':
#创建时间索引
    df = pd.DataFrame({"values": Train_sequence})

    # **自动选择最优 SARIMA (p, d, q) × (P, D, Q, s)**
    model_auto = auto_arima(df["values"], 
                            seasonal=True,  # 开启季节性
                            m=12,            # 设定周期（这里假设周期是 5）
                            trace=True,      # 显示优化过程
                            error_action="ignore", 
                            suppress_warnings=True, 
                            stepwise=True)   # 逐步搜索最佳参数
    
    print(f"\n最佳 SARIMA 参数: {model_auto.order} × {model_auto.seasonal_order}")  # 输出自动选择的 (p, d, q) × (P, D, Q, s)
    
    # **训练 SARIMA 模型**
    model = SARIMAX(df["values"], 
                    order=model_auto.order, 
                    seasonal_order=model_auto.seasonal_order)
    SARIMA_model = model.fit()
    
    # **滚动预测未来 5 个时间步**
    n_steps = steplength  # 预测未来 5 个时间点
    future_preds = SARIMA_model.forecast(steps=n_steps)
    
    #计算在测试集上的mse
    SARIMA_mse = np.mean((np.array(Test_sequence) - np.array(future_preds)) ** 2)
    #保存模型误差
    mse_dict['SARIMA'] = SARIMA_mse
    # 保存模型
    model_dict['SARIMA'] = SARIMA_model


##########################################################################################
'''Holt-Winters'''########################################################################
##########################################################################################
#只有在特征的时间分辨率为 月 的时候使用
if FeatureTimeResolution == 'M':
    #创建时间索引
    df = pd.DataFrame({'values': Train_sequence})

    # 训练 Holt-Winters 模型
    hw_model = ExponentialSmoothing(
        df["values"], 
        trend="add",        # 加性趋势
        seasonal="add",     # 加性季节性
        seasonal_periods=12 # 设定季节性周期为 12
    ).fit()

    # 预测未来 steplength 个时间点
    n_steps = steplength
    future_preds = hw_model.forecast(steps=n_steps)
    
    #计算在测试集上的mse
    hw_mse = np.mean((np.array(Test_sequence) - np.array(future_preds)) ** 2)
    #保存模型误差
    mse_dict['hw'] = hw_mse
    # 保存模型
    # model_dict['hw'] = hw_model


##########################################################################################
'''SES'''################################################################################
##########################################################################################
# 简单指数平滑
df = Train_sequence
SES_model = SimpleExpSmoothing(df).fit(optimized=True)
n_steps = steplength
future_preds = SES_model.forecast(steps=n_steps)

# 输出自动优化后的 α 值
print(f"自动优化的平滑系数 α: {SES_model.model.params['smoothing_level']:.4f}")

#计算在测试集上的mse
SES_mse = np.mean((np.array(Test_sequence) - np.array(future_preds)) ** 2)
#保存模型误差
mse_dict['SES'] = SES_mse
# 保存模型
# model_dict['SES'] = SES_model

##########################################################################################
##########################################################################################
##########################################################################################

# 获取值在测试集上误差最小的模型
min_key = min(mse_dict, key=mse_dict.get)

   
if min_key == 'ARIMA':
    #创建时间索引 
    df = pd.DataFrame({"values": sequence})

    # **自动选择 ARIMA (p, d, q) 参数**
    model_auto = auto_arima(df["values"], 
                            seasonal=False,  # 非季节性数据
                            trace=True,      # 显示优化过程
                            error_action="ignore", 
                            suppress_warnings=True, 
                            stepwise=True)   # 逐步搜索最佳参数

    print(f"\n最佳 ARIMA 参数: {model_auto.order}")  # 输出自动选择的 (p, d, q)

    # **训练 ARIMA 模型**
    model = ARIMA(df["values"], order=model_auto.order)
    ARIMA_model = model.fit()
    
    #获取残差序列
    residuals = ARIMA_model.resid

    # **预测未来 steplength 个时间步**
    n_steps = steplength  # 预测未来 steplength 个时间点
    future_preds = ARIMA_model.forecast(steps=n_steps)

elif min_key == 'hw':
    #创建时间索引
    df = pd.DataFrame({'values': sequence})

    # 训练 Holt-Winters 模型
    hw_model = ExponentialSmoothing(
        df["values"], 
        trend="add",        # 加性趋势
        seasonal="add",     # 加性季节性
        seasonal_periods=12 # 设定季节性周期为 12
    ).fit()
    
    # 获取拟合值
    fitted_values = hw_model.fittedvalues
    
    # 计算残差（真实值 - 预测值）
    residuals = df["values"] - fitted_values

    # 预测未来 steplength 个时间点
    n_steps = steplength
    future_preds = hw_model.forecast(steps=n_steps)
    
elif min_key == 'SARIMA':
    df = pd.DataFrame({"values": sequence})

    # **自动选择最优 SARIMA (p, d, q) × (P, D, Q, s)**
    model_auto = auto_arima(df["values"], 
                            seasonal=True,  # 开启季节性
                            m=12,            # 设定周期（这里假设周期是 5）
                            trace=True,      # 显示优化过程
                            error_action="ignore", 
                            suppress_warnings=True, 
                            stepwise=True)   # 逐步搜索最佳参数
    
    print(f"\n最佳 SARIMA 参数: {model_auto.order} × {model_auto.seasonal_order}")  # 输出自动选择的 (p, d, q) × (P, D, Q, s)
    
    # **训练 SARIMA 模型**
    model = SARIMAX(df["values"], 
                    order=model_auto.order, 
                    seasonal_order=model_auto.seasonal_order)
    SARIMA_model = model.fit()
    
    #获取残差序列
    residuals = SARIMA_model.resid
    
    
    # **预测未来 5 个时间步**
    n_steps = steplength  # 预测未来 5 个时间点
    future_preds = SARIMA_model.forecast(steps=n_steps)
    


else:
    # 简单指数平滑
    df = sequence
    SES_model = SimpleExpSmoothing(df).fit(optimized=True)
    
    # 获取拟合值
    fitted_values = SES_model.fittedvalues
    
    # 计算残差（真实值 - 预测值）
    residuals = pd.Series(df) - fitted_values
    
    n_steps = steplength
    future_preds = SES_model.forecast(steps=n_steps)


# Ljung-Box 检验（默认滞后阶数 lags=10）
ljung_box_result = acorr_ljungbox(residuals, lags=[10], return_df=True)
# 输出结果
print(ljung_box_result)



# 绘制残差时间序列图
plt.figure(figsize=(10, 4))
plt.plot(residuals, label="Residuals", color='blue')
plt.axhline(y=0, color='r', linestyle='--')  # 添加 0 参考线
plt.xlabel("Time")
plt.ylabel("Residuals")
plt.title("Residuals over Time")
plt.legend()
plt.show()























future_preds2 = [x*std_value+mean_value for x in future_preds]
future_preds3 = [np.exp(x)-1e-6 for x in future_preds2]
FutureData[element] = future_preds3

FutureDatadf = pd.DataFrame(FutureData)
FutureDatadf.index = FutureIndex
FutureDatadf.index.name = 'time'

FeatureFuture = pd.concat([data,FutureDatadf], axis=0)
#FeatureFuture.to_csv('.\\FeatureForFuture\\'+item+'.csv', index=True)
    


end_time = time.time()  # 记录结束时间       
print(f"程序运行时间: {end_time - start_time:.6f} 秒")   