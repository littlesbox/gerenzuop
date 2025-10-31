# -*- coding: utf-8 -*-
"""
为未来的预测构建特征
"""

import os
import pandas as pd
import numpy as np
import time

start_time = time.time()

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
                                
FeatureDir1 = '.\\FeatureForFuture'
sample_place = '.\\PAEsData\\sample_place_info.csv'

#设定时间点
FutureTime = '2027-4-30'

#获取每个特征文件的路径，包括文件名
files_path = list_files(FeatureDir1)
file_csv_path = []
#筛选出所有csv文件的路径，csv文件为特征文件
for item in files_path:
    houzhui = item.rsplit('.',1)
    if houzhui[1] == 'csv':
        file_csv_path.append(item)

#sample_place_info
sample_place_info = pd.read_csv(sample_place, sep=',') 

# #将特征文件的名字与其路径对应起来，建立一个字典，特征文件名不包含后缀
# file_csv_name = [os.path.basename(x)[:-4] for x in file_csv_path]
# FeatureFiles2Path = dict(zip(file_csv_name, file_csv_path))

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

traindata = []
for i in range(sample_place_info.shape[0]):
    #获取样本信息
    SampleNum = i + 1
    SampleLocation3C = sample_place_info.loc[i,'Location3C']
    SampleLocation4C = sample_place_info.loc[i,'Location4C']
    SamplePlace = sample_place_info.loc[i,'place']
    SampleTime = FutureTime
    
    #为当前样本构建特征
    FeatureValuedict = {}
    for item in FeatureInfo.index.tolist():
        #提取当前特征的数据
        data = FeatureName2Data[item]
        
        #获取当前特征信息
        FeatureName = item #这里的FeatureName不带后缀名
        FeatureTimeResolution = FeatureInfo.loc[FeatureName, 'TimeResolution']
        FeatureSumSence = FeatureInfo.loc[FeatureName, 'SumSence']
        FeatureSpatial = FeatureInfo.loc[FeatureName, 'Spatial'] #表示当前特征在空间上使用何种方式确定
        
        #将采样时间转换成datetime格式，并且提取其中的年份和月份
        #从而确定特征数据选取的时间段
        SampleTime = pd.to_datetime(SampleTime)
        SampleTime = SampleTime + pd.offsets.MonthEnd(0) #将采样时间精确到月份，日设定为当月最后一日
        SampleTimeYear = SampleTime.year
        SampleTimeMonth = SampleTime.month
        
        #根据当前的特征的时间分辨率，确定特征数据的取样时间段
        if FeatureTimeResolution == 'Y' :   #如果当前特征的时间分辨率是 年
            FeatureDataStart = str(SampleTimeYear-1) + '-12-31'
            FeatureDataend = str(SampleTimeYear) + '-12-31'
                        
            #判断当前特征是否具有可加意义,据此来构建指标
            if FeatureSumSence == 1: #如果具有可加意义
                colname = sample_place_info.loc[i,FeatureSpatial] #取出当前样本的空间位置
                colname = str(colname) #样本的空间位置可能是数字，例如在SMroot中
                FeatureValue1 = data.loc[FeatureDataStart,colname]
                FeatureValue2 = data.loc[FeatureDataend,colname]
                w = SampleTimeMonth/12
                FeatureValue = FeatureValue1 * (1-w) + FeatureValue2 * w
                FeatureValuedict[item] = FeatureValue
            else:
                colname = sample_place_info.loc[i,FeatureSpatial] #取出当前样本的空间位置
                FeatureValue = data.loc[FeatureDataend,colname]
                FeatureValuedict[item+'_mean'] = FeatureValue
                FeatureValuedict[item+'_td'] = 0
                
        elif FeatureTimeResolution == 'M':
            FeatureDataStart = SampleTime - pd.DateOffset(months=11)
            timeperoid = pd.date_range(start=FeatureDataStart, periods=12, freq="ME")
            timeperoid = timeperoid.strftime("%Y-%m-%d").tolist() #将日期转成字符串，用于索引
            
            #判断当前特征是否具有可加意义,据此来构建指标
            if FeatureSumSence == 1: #如果具有可加意义
                colname = sample_place_info.loc[i,FeatureSpatial] #取出当前样本的空间位置
                colname = str(colname) #样本的空间位置可能是数字，例如在SMroot中
                FeatureValuelist = data.loc[timeperoid,colname].tolist()
                FeatureValuelist = np.array(FeatureValuelist)
                FeatureValuedict[item] = np.sum(FeatureValuelist)
            else:
                colname = sample_place_info.loc[i,FeatureSpatial] #取出当前样本的空间位置
                colname = str(colname) #样本的空间位置可能是数字，例如在SMroot中
                FeatureValuelist = data.loc[timeperoid,colname].tolist()
                FeatureValuelist = np.array(FeatureValuelist)
                FeatureValuedict[item+'_mean'] = np.mean(FeatureValuelist)
                FeatureValuedict[item+'_td'] = np.var(FeatureValuelist,ddof=1)
            
        else:
            timeperoid = 0
            colname = sample_place_info.loc[i,FeatureSpatial] #取出当前样本的空间位置
            colname = str(colname) #样本的空间位置可能是数字，例如在SMroot中
            FeatureValue = data.loc[timeperoid,colname]
            FeatureValuedict[item] = FeatureValue
        
    traindata.append(FeatureValuedict)
        
traindata_df = pd.DataFrame(traindata)
traindata_df.insert(0, 'place', sample_place_info['place'])
#traindata_df.dropna(inplace=True)

traindata_df1 = traindata_df.copy()
traindata_df1['PAEsType'] = 0
#traindata_df1['PAEsConc'] = PAEsData['DnBP']

traindata_df2 = traindata_df.copy()
traindata_df2['PAEsType'] = 1
#traindata_df2['PAEsConc'] = PAEsData['DEHP']

traindata_df3 = pd.concat([traindata_df1,traindata_df2])

#删除含有nan的行
#traindata_df3.dropna(inplace=True)
#删除最后一列为-1的行
#traindata_df3.drop(traindata_df3[traindata_df3["PAEsConc"] <= 0].index, inplace=True)


traindata_df3.to_excel('traindata_Future.xlsx', index=False)
traindata_df3.to_csv('traindata_Future.csv', index=False)
        
end_time = time.time()  # 记录结束时间       
print(f"程序运行时间: {end_time - start_time:.6f} 秒")    
        
