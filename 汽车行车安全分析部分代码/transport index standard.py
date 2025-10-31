import pandas as pd

#设置路径获取文件名列表
p="F:\\工程\\大创\\data\\transport index.csv"

#读取文件并转换成列表
transportindex=pd.read_csv(p, sep=',')
transportindexT = pd.DataFrame(transportindex.values.T, index=transportindex.columns, columns=transportindex.index)
transportindexlist=transportindexT.values.tolist()
transportliststand=[]
standardindex=[]

transportliststand.append(transportindexlist[0])
for i in range(1,len(transportindexlist)):
	ma=max(transportindexlist[i])
	mi=min(transportindexlist[i])

	if ma != mi:
		for item in transportindexlist[i]:
			index=(item-mi)/(ma-mi)
			standardindex.append(index)
	else:
		for item in transportindexlist[i]:
			standardindex.append(item)

	transportliststand.append(standardindex)
	standardindex=[]

index_files=pd.DataFrame(transportliststand)
transportindexstandard = pd.DataFrame(index_files.values.T, index=transportindex.index, columns=transportindex.columns)
transportindexstandard.to_csv("F:\\工程\\大创\\data\\transport index standard.csv",index=False,encoding='utf_8_sig')