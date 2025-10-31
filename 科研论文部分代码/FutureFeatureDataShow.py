import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import numpy as np

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


path = '.\\FeatureForFuture\\'
name = 'Temperature'
place = '9'

df = pd.read_csv(path+name+'.csv',index_col=0)
df1= df.iloc[:-52,:]
df2 = df.iloc[-52:,:]



# 创建数据
time = np.arange(1, df.shape[0]+1)  # 横轴时间 1-10
value = df[place]  # 纵轴值

# 设定刻度位置（起始点、中间分界点、结尾点）
tick_positions = [time[0], time[-52], time[-1]]  
tick_labels = ['1994-01', '2023-01', '2027-04']  # 自定义刻度标签

# 创建画布
plt.figure(figsize=(8, 5))

# 绘制折线（不同颜色）
plt.plot(time[:-52], value[:-52], linestyle='-', color='#1f77b4', linewidth=1, label='actual')
plt.plot(time[-52:], value[-52:], linestyle='-', color='#ff7f0e', linewidth=1, label='predict')

# 设置 x 轴刻度
plt.xticks(tick_positions, tick_labels)  

# 添加轴标签
plt.xlabel('time')
plt.ylabel('value')

# 添加标题和图例
plt.title(name+'_'+place)
plt.legend()

plt.savefig(path+'Figure\\'+name+'_'+place+'.png',bbox_inches='tight')

# 显示图像
plt.show()


