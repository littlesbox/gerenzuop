import os
import os.path

#读取文件夹文件，并将文件名与扩展名分开
p='F:\\班委工作文件\\小论文 实变函数\\photo\\'
homework=os.listdir(p)
workname=[]
worknameex=[]
for item in homework:
	tup=os.path.splitext(item)
	workname.append(tup[0])
	worknameex.append(tup[1])						

#读取正确命名格式,不包含扩展名，正确命名格式为 “学号 姓名”
with open('F:\\班委工作文件\\170501花名册\\workname_format.txt') as f:
	t=f.readlines()
correct_name=[]
for item in t:
	correct_name.append(item[:-1])

#获取学生名单
unsubmit=[item[11:] for item in correct_name]

chongfu=[]
count={}
for item in unsubmit:
	count[item]=0
	for element in workname:
		if item in element:
			count[item]=count[item]+1

for item in count:
	if count[item] > 1:
		chongfu.append(item)


if chongfu:
	print('有人重复提交，重复的人为：')
	print(chongfu)
else:
	#逐个判断已提交的作业是否命名正确
	for i in range(len(workname)):
		if workname[i] in correct_name: 				
			unsubmit.remove(workname[i][11:]) #如果提交了且命名格式正确就将其从未提交名单中删去
			continue
		else: #如果命名不正确,就找出是谁的作业然后重命名，并将其从未提交名单中删去
			for item in correct_name:
				if item[11:] in workname[i]:
					unsubmit.remove(item[11:])
					os.rename(p+'\\'+workname[i]+worknameex[i],p+'\\'+item+worknameex[i])
	
	if unsubmit:
		print('有人没交，没交的人为：')
		print(unsubmit)
	#	n=str(len(unsubmit))
	#	unsubmit='\n'.join(unsubmit)
	#	with open(p+'\\'+'unsubmit.txt','w') as f:
	#		f.write(unsubmit)
	#		f.write('\n共'+n+'人未交')
	else:
	#	os.remove(p+'\\'+'unsubmit.txt')
		print('已收齐')