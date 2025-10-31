'''
统计每个模型的最后一次的训练结果，结果指的是相关的性能指标
'''

import os
import pandas as pd
from collections import Counter

#筛选出所有保存模型训练结果的文件夹，将文件夹的名字保存在列表content中
content = []
subdirectories = os.listdir()
for item in subdirectories:
    # 判断目标路径是不是文件夹
    if os.path.isdir(item):
        # 如果是文件夹就存入列表
        content.append(item)
        

# 将这些文件夹按照模型的种类进行分类
# model_type用于存放模型的种类
# 将保存训练结果的文件夹的名称按照一定的格式进行分裂，用于确定该文件夹所属的模型种类
# content_splited用与存放分裂后的文件夹名称，其中每个元素是一个列表
# split_for为分裂的标准
model_type = []
content_splited = []
split_for = '-'

for item in content:
   splited_name  = item.split(split_for)
   model_type.append(splited_name[0])
   content_splited.append(splited_name)
   
model_type_count = Counter(model_type)
model_type_count = dict(model_type_count)
model_type = model_type_count.keys()
model_type = list(model_type)
model_type.sort()


#将保存训练结果的文件夹按照模型种类进行分类，每个种类存入一个列表，最后再将这些列表存入model_class_list中
model_class_list = []

for item in model_type:
    temp = []
    #将模型种类为item的文件夹全部放入temp列表中
    for element in content_splited:     
        if item in element:      #判断element这个文件夹对应的模型种类是不是item这个类型
            temp.append(element)
    model_class_list.append(temp)   #将item种类的文件夹做成的列表放入model_class_list中

#对分好类的结果文件夹进行时间上的识别，将文件夹创建的时间作为键，文件夹的名称作为值，生成一个字典
#每个字典对应着一个模型种类的结果文件夹，即将同一个模型种类的结果文件夹放入同一个字典中，时间作为键，文件夹的名称作为值
#将得到这些字典存入model_class列表中
model_class = []    
for item in model_class_list:
    temp = {}
    for element in item:
        time = int(''.join(element[1:]))
        temp[time] = split_for.join(element)
    model_class.append(temp)

#在每个模型种类的结果文件夹中寻找最后一次创建的那个，并且将其放入directory中
directory = []
for item in model_class:
    item_key = list(item.keys())
    maxkey = max(item_key)
    directory.append(item[maxkey])

#读取结果文件中的性能指标文件
performance_list = []
for item in directory:
    data = pd.read_csv(os.path.join(item,'assess_list.csv'), sep=',', header=None)
    data = data.values.tolist()
    temp = {}
    for element in data:
        temp[element[0]] = element[1]
    temp1 = [item, temp['R2'], temp['MSE']]
    performance_list.append(temp1)
    
performance = pd.DataFrame(performance_list)
performance.columns = ['model', 'R2', 'MSE']
performance.to_csv('00permance_last_time.csv', index=None)