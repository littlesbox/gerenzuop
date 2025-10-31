%%
%附件3数据读取
clear;
point=xlsread('附件3：长春市9个区交通网络数据和主要小区相关数据.xlsx','交通路口节点数据'); 
community=xlsread('附件3：长春市9个区交通网络数据和主要小区相关数据.xlsx','各区主要小区数据');
point=[point,zeros(size(point,1),1)];
%%
point_num=length(point(:,1)); %路口数
community_num=length(community(:,1)); %小区数
D=zeros(point_num,community_num);
%%
%计算每个路口与小区的距离
for i=1:point_num
    for j=1:community_num
        D(i,j)=pdist([point(i,2) point(i,3);community(j,5) community(j,6)],'euclidean');
    end
end
%%
%为每个路口匹配城市
for i=1:point_num
    [~,index]=min(D(i,:));
    point(i,4)=community(index,9);
end
%%
%搜索距离小区最近的路口
for i=1:community_num
    [~,index]=min(D(:,i));
    community(i,8)=index;
end
community(:,7)=[];

save dis_piont2community