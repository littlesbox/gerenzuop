%根据路径信息创建路口间的可达矩阵
%%
%附件3数据读取
clear;
point=xlsread('附件3：长春市9个区交通网络数据和主要小区相关数据.xlsx','交通路口节点数据'); 
dis=xlsread('附件3：长春市9个区交通网络数据和主要小区相关数据.xlsx','交通路口路线数据'); 
%%
%创建路口间的可达矩阵
point_num=length(point(:,1)); %路口数
path=Inf*ones(point_num);
for i=1:point_num
    path(i,i)=0;
end
path_num=length(dis(:,1)); %路径数
temp=0;
for i=1:path_num
    path(dis(i,2),dis(i,3))=dis(i,4);
    path(dis(i,3),dis(i,2))=dis(i,4);
end

save('path','path')