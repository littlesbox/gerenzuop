'''
统计每个模型的所有的训练结果，结果指的是相关的性能指标
'''

import os
import pandas as pd


#筛选出所有保存模型训练结果的文件夹，将文件夹的名字保存在列表content中
content = []
subdirectories = os.listdir()
for item in subdirectories:
    # 判断目标路径是不是文件夹
    if os.path.isdir(item):
        # 如果是文件夹就存入列表
        content.append(item)
        
        
#读取结果文件中的性能指标文件
performance_list = []
for item in content:
    data = pd.read_csv(os.path.join(item,'assess_list.csv'), sep=',', header=None)
    data = data.values.tolist()
    temp = {}
    for element in data:
        temp[element[0]] = element[1]
    temp1 = [item, temp['R2'], temp['MSE'],temp['MSE'],temp['MSE']]
    performance_list.append(temp1)
    
performance = pd.DataFrame(performance_list)
performance.columns = ['model', 'R2', 'MSE']
performance.to_csv('00permance_all_time.csv', index=None)