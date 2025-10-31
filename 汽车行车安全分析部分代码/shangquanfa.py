import pandas as pd
import math

#设置路径获取文件
p="F:\\工程\\大创\\data\\transport index standard.csv"

#读取文件并转换成列表
transport=pd.read_csv(p, sep=',')
transportT=pd.DataFrame(transport.values.T, index=transport.columns, columns=transport.index)
transportlist=transportT.values.tolist()
n=len(transportlist[1])
count={}
e=[]
d=[]
sume=0
sumd=0

for i in range(1,len(transportlist)):
	for item in transportlist[i]:
		if item != 0:
			sume=sume+(item/sum(transportlist[i]))*math.log(item/sum(transportlist[i]))

	sume=(-sume)/math.log(n)
	sumd=1-sume

	e.append(sume)
	d.append(sumd)

	sume=0
	sumd=0

info=[e,d]
info_file=pd.DataFrame(info,columns=list(transport.columns)[1:])
info_file.to_csv("F:\\工程\\大创\\data\\info_file.csv",index=False,encoding='utf_8_sig',sep=',')