import os
import pandas as pd
import time

#设置路径获取文件名列表
p="F:\\工程\\大创\\data\\transport data part\\"
files_name=os.listdir(p)

transportindex=[]
header0=['车牌号']
header1=['急加速次数','急加速时长','急加速数量比例','急加速时长占比']
header2=['急减速次数','急减速时长','急减速数量比例','急减速时长占比']
header3=['急转向次数','急转向时长','急转向数量比例','急转向时长占比']
header4=['超速次数','超速时长','超速数量比例','超速时长占比']
header5=['熄火滑行次数','熄火滑行时长','熄火滑行数量比例','熄火滑行时长占比']
header6=['疲劳驾驶次数','疲劳驾驶时长','处于疲劳状态驾驶时长','疲劳驾驶时长占比','处于疲劳状态驾驶时长占比']
header=header0+header1+header2+header3+header4+header5+header6


#遍历文件
for element in files_name:
	#读取文件并转换成列表
	car_data=pd.read_csv(p+element, sep=',')
	car_datalist=car_data.values.tolist()


	#转换时间格式为时间戳，转换速度单位 14:timeatamp 15:nationalspeed
	for item in car_datalist:
		timearray = time.strptime(item[10], "%Y-%m-%d %H:%M:%S") 
		timestamp = int(time.mktime(timearray))
		nationalspeed=item[11]/3.6
		item.append(timestamp)
		item.append(nationalspeed)


	#急加、减速 急转弯 超速 总驾驶时间
	#设置指标变量
	rapidatimes=0
	rapidakeep=0
	rapiddetimes=0
	rapiddekeep=0
	rapidturntimes=0
	rapidturnkeep=0
	hypervtimes=0
	hypervkeep=0
	drivekeep1=0

	#指标计算
	#设定初始时间点，结束时间点
	begin=car_datalist[0][13]
	end=car_datalist[-1][13]

	#设置时间断层变量
	discontinue=0

	#循环计算
	for i in range(len(car_datalist)-1):
		nowtime=car_datalist[i][13]
		nexttime=car_datalist[i+1][13]
		if nowtime == nexttime:
			continue
		#判断时间是否断层
		if nexttime-nowtime > 5:
			#如果断层则计算上一层的总时长，并重置开始时间和结束时间为下一层
			drivekeep1=drivekeep1+(nowtime-begin)
			begin=nexttime
			discontinue=1
		else:
			#没有断层则开始计算
			#计算加速度，并判定是否为急加减速
			a=(car_datalist[i+1][14]-car_datalist[i][14])/(nexttime-nowtime)
			if a > 3:
				rapidatimes=rapidatimes+1
				rapidakeep=rapidakeep+(nexttime-nowtime)
			elif a < -3:
				rapiddetimes=rapiddetimes+1
				rapiddekeep=rapiddekeep+(nexttime-nowtime)
			
			#计算转过的角度，并处理越过0度线的情况
			turnangle=car_datalist[i+1][2]-car_datalist[i][2]
			if turnangle >= 180:
				turnangle=turnangle-360
			elif turnangle <= -180:
				turnangle=turnangle+360
			
			#计算角速度，并判定是否为急转向
			omega=turnangle/(nexttime-nowtime)
			if car_datalist[i][11] > 50 and abs(omega) > 10:
				rapidturntimes=rapidturntimes+1
				rapidturnkeep=rapidturnkeep+(nexttime-nowtime)

			#判断当前是否超速
			if car_datalist[i][11] > 60:
				hypervtimes=hypervtimes+1
				if car_datalist[i+1][11] > 60:
					hypervkeep=hypervkeep+(nexttime-nowtime)

	if car_datalist[-1][11] > 60:
				hypervtimes=hypervtimes+1

	#如果不存在时间断层
	if discontinue == 0:
		drivekeep1=car_datalist[-1][13]-car_datalist[0][13]
	else:
		drivekeep1=drivekeep1+(end-begin)
	#重置时间断层变量
	discontinue=0


	#熄火滑行
	#设置指标变量
	flameoutsliptimes=0
	flameoutslipkeep=0
	drivekeep2=0

	#指标计算
	#设定初始时间点，结束时间点
	begin=car_datalist[0][13]
	end=car_datalist[-1][13]

	#循环计算
	for i in range(len(car_datalist)-1):
		nowtime=car_datalist[i][13]
		nexttime=car_datalist[i+1][13]
		if nowtime == nexttime:
			continue
		#判断时间是否断层
		if nexttime-nowtime > 90:
			#如果断层则计算上一层的总时长，并重置开始时间和结束时间为下一层
			drivekeep2=drivekeep2+(nowtime-begin)
			begin=nexttime
			discontinue=1
		else:
			#没有断层则开始计算
			if car_datalist[i][5] == 0 and car_datalist[i][11] != 0:
				flameoutsliptimes=flameoutsliptimes+1
				if car_datalist[i+1][5] == 0 and car_datalist[i+1][11] != 0:
					flameoutslipkeep=flameoutslipkeep+(nexttime-nowtime)
	if car_datalist[-1][5] == 0 and car_datalist[-1][11] != 0:
				flameoutsliptimes=flameoutsliptimes+1

	#如果不存在时间断层
	if discontinue == 0:
		drivekeep2=car_datalist[-1][13]-car_datalist[0][13]
	else:
		drivekeep2=drivekeep2+(end-begin)
	#重置时间断层变量
	discontinue=0



	#疲劳驾驶
	#获取连续驾驶行为信息
	#设置驾驶行为变量
	drive=[]
	drivestate=[]
	drivekeep=0
	drivekeep3=0
	#最后一层是否有连续驾驶行为指示变量
	lastfloor=1

	#寻找开始行车的初始时间点
	for i in range(len(car_datalist)):
		if car_datalist[i][5] == 1 or car_datalist[i][11] != 0:
			drivebegin=car_datalist[i][13]
			driveend=car_datalist[i][13]
			start=i
			break

	#开始获取连续驾驶行为信息
	for i in range(start,len(car_datalist)-1):
		nowtime=car_datalist[i][13]
		nexttime=car_datalist[i+1][13]
		if nowtime == nexttime or nowtime < drivebegin:
			continue
		#判断时间是否断层
		if nexttime-nowtime > 300:
			if car_datalist[i][5] == 1 or car_datalist[i][11] != 0:
				driveend=car_datalist[i][13]
				drivekeep=driveend-drivebegin
				drivestate.append(drivebegin)
				drivestate.append(driveend)
				drivestate.append(drivekeep)
				drive.append(drivestate)
				drivestate=[]
			else:
				drivestate.append(drivebegin)
				drivestate.append(driveend)
				drivestate.append(drivekeep)
				drive.append(drivestate)
				drivestate=[]
			for j in range(i+1,len(car_datalist)):
				if car_datalist[j][5] == 1 or car_datalist[j][11] != 0:
					drivebegin=car_datalist[j][13]
					driveend=car_datalist[j][13]
					break
				elif j == len(car_datalist)-1:
					lastfloor=0
		else:
			#如果没有断层
			if car_datalist[i][5] == 1 or car_datalist[i][11] != 0:
				driveend=car_datalist[i][13]
				drivekeep=driveend-drivebegin
			else:
				if car_datalist[i+1][5] == 1 or car_datalist[i+1][11] != 0:
					drivestate.append(drivebegin)
					drivestate.append(driveend)
					drivestate.append(drivekeep)
					drive.append(drivestate)
					drivestate=[]
					drivebegin=car_datalist[i+1][13]
					driveend=car_datalist[i+1][13]	
				else:
					continue

	#判断最后一层有没有连续驾驶行为
	if lastfloor != 0:
		if car_datalist[-1][5] == 1 or car_datalist[-1][11] != 0:
			driveend=car_datalist[-1][13]
			drivekeep=driveend-drivebegin
			drivestate.append(drivebegin)
			drivestate.append(driveend)
			drivestate.append(drivekeep)
			drive.append(drivestate)
			drivestate=[]
		else:
			drivestate.append(drivebegin)
			drivestate.append(driveend)
			drivestate.append(drivekeep)
			drive.append(drivestate)
			drivestate=[]


	#计算驾驶时长
	#设定初始时间点，结束时间点
	begin=car_datalist[0][13]
	end=car_datalist[-1][13]

	#计算驾驶时间，时间断层阈值设为5分钟
	for i in range(len(car_datalist)-1):
		nowtime=car_datalist[i][13]
		nexttime=car_datalist[i+1][13]
		if nowtime == nexttime:
			continue
		#判断时间是否断层
		if nexttime-nowtime > 300:
			#如果断层则计算上一层的总时长，并重置开始时间和结束时间为下一层
			drivekeep3=drivekeep3+(nowtime-begin)
			begin=nexttime
			discontinue=1

	#如果不存在时间断层
	if discontinue == 0:
		drivekeep3=car_datalist[-1][13]-car_datalist[0][13]
	else:
		drivekeep3=drivekeep3+(end-begin)
	#重置时间断层变量
	discontinue=0

	#计算两段连续驾驶行为的时间差
	driveinterval=[]
	for i in range(len(drive)-1):
		interval=drive[i+1][0]-drive[i][1]
		driveinterval.append(interval)


	#疲劳驾驶行为识别
	#识别并获取四小时疲劳驾驶信息
	subdrive=[]
	subdrivekeep=0
	start=0
	suminterval=0

	for i in range(len(driveinterval)):
		suminterval=suminterval+driveinterval[i]
		if suminterval >= 1200:
			for j in range(start,i+1):
				subdrivekeep=subdrivekeep+drive[j][2]
			subdrive.append(subdrivekeep)
			subdrivekeep=0
			start=i+1
			suminterval=0

	for j in range(start,len(drive)):
		subdrivekeep=subdrivekeep+drive[j][2]
	subdrive.append(subdrivekeep)
	subdrivekeep=0

	#获取4小时疲劳驾驶信息次数，时长
	count4=0
	triedkeep4=0
	if subdrive:
		for item in subdrive:
			if item >= 14400:
				count4=count4+1
				triedkeep4=triedkeep4+item

	#识别并获取八小时疲劳驾驶信息
	subdrive8=[]
	subdrivekeep=0
	start=0
	keeptime=0

	for i in range(len(drive)):
		keeptime=drive[i][1]-drive[start][0]
		if keeptime > 86400:
			for j in range(start,i):
				subdrivekeep=subdrivekeep+drive[j][2]
			subdrive8.append(subdrivekeep)
			subdrivekeep=0
			start=i
			keeptime=0

	for j in range(start,len(drive)):
		subdrivekeep=subdrivekeep+drive[j][2]
	subdrive8.append(subdrivekeep)
	subdrivekeep=0

	#获取8小时疲劳驾驶信息次数，时长
	count8=0
	triedkeep8=0
	if subdrive8:
		for item in subdrive8:
			if item >= 28800:
				count8=count8+1
				triedkeep8=triedkeep8+item


	#整理指标
	#疲劳驾驶指标 次数 持续时间 处于疲劳状态的驾驶时间 及它们的比例
	triedcount=count8+count4
	triedkeep=triedkeep8+triedkeep4
	intriedkeep=(triedkeep8-count8*8*3600)+(triedkeep4-count4*4*3600)
	triedkeeprad=triedkeep/drivekeep3
	intriedkeeprad=intriedkeep/drivekeep3
	tried=[triedcount,triedkeep,intriedkeep,triedkeeprad,intriedkeeprad]

	#急加速 急减速 急转向 超速 的时间比与数量比
	rapidatimesrad=rapidatimes/(len(car_datalist)-1)
	rapidakeeprad=rapidakeep/drivekeep1
	rapiddetimesrad=rapiddetimes/(len(car_datalist)-1)
	rapiddekeeprad=rapiddekeep/drivekeep1
	rapidturntimesrad=rapidturntimes/(len(car_datalist)-1)
	rapidturnkeeprad=rapidturnkeep/drivekeep1
	hypervtimesrad=hypervtimes/len(car_datalist)
	hypervkeeprad=hypervkeep/drivekeep1

	rapida=[rapidatimes,rapidakeep,rapidatimesrad,rapidakeeprad]
	rapidde=[rapiddetimes,rapiddekeep,rapiddetimesrad,rapiddekeeprad]
	rapidturn=[rapidturntimes,rapidturnkeep,rapidturntimesrad,rapidturnkeeprad]
	hyperv=[hypervtimes,hypervkeep,hypervtimesrad,hypervkeeprad]

	#熄火滑行时间比与数量比
	flameoutsliptimesrad=flameoutsliptimes/len(car_datalist)
	flameoutslipkeeprad=flameoutslipkeep/drivekeep2

	flameoutslip=[flameoutsliptimes,flameoutslipkeep,flameoutsliptimesrad,flameoutslipkeeprad]

	index=[element[:-4]]+rapida+rapidde+rapidturn+hyperv+flameoutslip+tried

	transportindex.append(index)

index_files=pd.DataFrame(transportindex,columns=header)
index_files.to_csv("F:\\工程\\大创\\data\\transport index.csv",index=False,encoding='utf_8_sig')