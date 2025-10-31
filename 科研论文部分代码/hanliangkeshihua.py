import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

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

PAEsFutureConc = pd.read_csv('PAEsFutureConc.csv')
PAEsData_OK = pd.read_csv('PAEsData_OK.csv')
PAEsDataConcNOyichang = pd.read_csv('PAEsConcNOyichang.csv')
xy = pd.read_csv('xy.csv')


# PAEsFutureConc_DnBP = PAEsFutureConc[PAEsFutureConc['PAEsType']==0]
# PAEsFutureConc_DEHP = PAEsFutureConc[PAEsFutureConc['PAEsType']==1]


PAEsDataConcNOyichang[PAEsDataConcNOyichang['num']==1]



zuobiao = []
for i in range(PAEsDataConcNOyichang.shape[0]):
    num = PAEsDataConcNOyichang.iloc[i,0]
    longitude = PAEsData_OK[PAEsData_OK['num']==num]['longitude'].iloc[0]
    latitude = PAEsData_OK[PAEsData_OK['num']==num]['latitude'].iloc[0]
    zuobiao.append([longitude,latitude])
    
zuobiaodf = pd.DataFrame(zuobiao)
zuobiaodf.columns = ['longitude','latitude']

PAEsDataConcNOyichangOK = pd.concat([PAEsDataConcNOyichang,zuobiaodf],axis=1)
PAEsDataConcNOyichangOK_DnBP = PAEsDataConcNOyichangOK[PAEsDataConcNOyichangOK['PAEsType']==0]
PAEsDataConcNOyichangOK_DEHP = PAEsDataConcNOyichangOK[PAEsDataConcNOyichangOK['PAEsType']==1]




zuobiao2 = []
for i in range(PAEsFutureConc.shape[0]):
    num = PAEsFutureConc.iloc[i,0]
    longitude = xy.iloc[num,0]
    latitude = xy.iloc[num,1]
    zuobiao2.append([longitude,latitude])
    
zuobiaodf2 = pd.DataFrame(zuobiao2)
zuobiaodf2.columns = ['longitude','latitude']

PAEsFutureConcOK = pd.concat([PAEsFutureConc,zuobiaodf2],axis=1)
PAEsFutureConcOK_DnBP = PAEsFutureConcOK[PAEsFutureConcOK['PAEsType']==0]
PAEsFutureConcOK_DEHP = PAEsFutureConcOK[PAEsFutureConcOK['PAEsType']==1]

##########################################################################################


fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

sc = axes[0].scatter(PAEsDataConcNOyichangOK_DnBP['longitude'], PAEsDataConcNOyichangOK_DnBP['latitude'], 
                     c=PAEsDataConcNOyichangOK_DnBP['PAEsConc'], 
                     cmap='coolwarm', edgecolor='k',s=100)

axes[0].set_title("DnBP_Past_Conc")
axes[0].set_xlabel("longitude")
axes[0].set_ylabel("latitude")



sc = axes[1].scatter(PAEsFutureConcOK_DnBP['longitude'], PAEsFutureConcOK_DnBP['latitude'], 
                     c=PAEsFutureConcOK_DnBP['PAEsConc'],
                     cmap='coolwarm', edgecolor='k',s=100)

axes[1].set_title("DnBP_Future_Conc")
axes[1].set_xlabel("longitude")
axes[1].set_ylabel("latitude")



# 共享颜色条
cbar = fig.colorbar(sc, ax=axes, orientation='vertical', fraction=0.05, pad=0.02)
cbar.set_label("values")

plt.subplots_adjust(right=0.85)

plt.savefig('DnBPConcCompare.png',bbox_inches='tight')

# 显示图像
plt.tight_layout()
plt.show()

###############################################################################################

fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

sc = axes2[0].scatter(PAEsDataConcNOyichangOK_DEHP['longitude'], PAEsDataConcNOyichangOK_DEHP['latitude'], 
                     c=PAEsDataConcNOyichangOK_DEHP['PAEsConc'], 
                     cmap='coolwarm', edgecolor='k',s=100)

axes2[0].set_title("DEHP_Past_Conc")
axes2[0].set_xlabel("longitude")
axes2[0].set_ylabel("latitude")



sc = axes2[1].scatter(PAEsFutureConcOK_DEHP['longitude'], PAEsFutureConcOK_DEHP['latitude'], 
                     c=PAEsFutureConcOK_DEHP['PAEsConc'],
                     cmap='coolwarm', edgecolor='k',s=100)

axes2[1].set_title("DEHP_Future_Conc")
axes2[1].set_xlabel("longitude")
axes2[1].set_ylabel("latitude")



# 共享颜色条
cbar = fig2.colorbar(sc, ax=axes2, orientation='vertical', fraction=0.05, pad=0.02)
cbar.set_label("values")

plt.subplots_adjust(right=0.85)

plt.savefig('DEHPConcCompare.png',bbox_inches='tight')

# 显示图像
plt.tight_layout()
plt.show()

#####################################################################################################

fig3, axes3 = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

sc = axes3[0].scatter(PAEsDataConcNOyichangOK_DnBP['longitude'], PAEsDataConcNOyichangOK_DnBP['latitude'], 
                     c=PAEsDataConcNOyichangOK_DnBP['PAEsConc'], 
                     cmap='coolwarm', edgecolor='k',s=100)

axes3[0].set_title("DnBP_Conc")
axes3[0].set_xlabel("longitude")
axes3[0].set_ylabel("latitude")



sc = axes3[1].scatter(PAEsDataConcNOyichangOK_DEHP['longitude'], PAEsDataConcNOyichangOK_DEHP['latitude'], 
                     c=PAEsDataConcNOyichangOK_DEHP['PAEsConc'],
                     cmap='coolwarm', edgecolor='k',s=100)

axes3[1].set_title("DEHP_Conc")
axes3[1].set_xlabel("longitude")
axes3[1].set_ylabel("latitude")



# 共享颜色条
cbar = fig3.colorbar(sc, ax=axes3, orientation='vertical', fraction=0.05, pad=0.02)
cbar.set_label("values")

plt.subplots_adjust(right=0.85)

plt.savefig('DnBP and DEHP Conc past.png',bbox_inches='tight')

# 显示图像
plt.tight_layout()
plt.show()