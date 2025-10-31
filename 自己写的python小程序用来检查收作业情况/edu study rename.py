import os
import os.path

#读取文件夹文件，并将文件名与扩展名分开
pforder='F:\\班委工作文件\\教育研究方法\\结课材料\\'
homeworkforder=os.listdir(pforder)
#print(homeworkforder)

#读取正确命名格式,不包含扩展名，正确命名格式为 “学号 姓名”
with open('F:\\班委工作文件\\教育研究方法\\format.txt') as f:
	t=f.readlines()
correct_name=[]
for item in t:
	correct_name.append(item[:-1])
#print(correct_name)

for item in homeworkforder:
	j=int(item[:2])-1
	
	p='F:\\班委工作文件\\教育研究方法\\结课材料\\'+item+'\\'
	homework=os.listdir(p)
	workname=[]
	worknameex=[]

	for ele in homework:
		tup=os.path.splitext(ele)
		workname.append(tup[0])
		worknameex.append(tup[1])

	for i in range(len(workname)):
		if 'ppt' in worknameex[i]: 				
			os.rename(p+workname[i]+worknameex[i],p+correct_name[j]+worknameex[i])
		elif '论文' in workname[i]:
			os.rename(p+workname[i]+worknameex[i],p+correct_name[j]+'(论文)'+worknameex[i])
		elif '开题' in workname[i]:
			os.rename(p+workname[i]+worknameex[i],p+correct_name[j]+'(开题报告)'+worknameex[i])

	os.rename(pforder+'\\'+item,pforder+'\\'+correct_name[j])