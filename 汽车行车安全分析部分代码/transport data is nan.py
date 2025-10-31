import os
import pandas as pd

p="F:\\工程\\大创\\data\\transport data\\"
files_name=os.listdir(p)

nanpart=[]
for item in files_name:
	car_data=pd.read_csv(p+item, sep=',')
	#print(car_data)
	totalnan=car_data.isnull().any().sum()
	if totalnan != 0:
		nanpart.append(item)
	else:
		nanpart.append('ok')
print(nanpart)