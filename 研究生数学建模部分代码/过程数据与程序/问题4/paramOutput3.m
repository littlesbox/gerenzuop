clear;
load('changchun.mat');
load('SPR.mat', 'P');
S_to=load('SPR.mat', 'S');
S_total=S_to.S;
DeliveryPoint=point(logical(bestindividual),1:3);
DeliveryPoint_num=sum(bestindividual);
[community_num,~]=size(community);
serve_R=zeros(community_num,1); %小区被服务半径
Delivery_num=zeros(DeliveryPoint_num,1); %投放点服务小区数
DeliveryPointServe_R=zeros(DeliveryPoint_num,1); %投放点服务半径
%%
%确定服务半径
D=D(logical(bestindividual),:);
tempServeR=max(min(D));
%%
serveCommunity=D<=tempServeR;
for i=1:DeliveryPoint_num
    Delivery_num(i)=sum(serveCommunity(i,:));  %确定服务小区量
    if Delivery_num(i)~=0
        DeliveryPointServe_R(i)=max(D(i,serveCommunity(i,:)));  %更新服务半径
    end
end
%%
%确定服务人数
S=S(logical(bestindividual),:);
serve_people=zeros(size(D));
for i=1:community_num
    distance=S(serveCommunity(:,i),i);
    distanceTotal=sum(distance);
    weight=distance/distanceTotal;
    [~,ind]=sort(distance);
    descWeight=sort(weight,'descend');
    weight(ind)=descWeight; %按照距离分配权重，距离越近，权重越大
    servePeople=weight*community(i,4);
    serve_people(serveCommunity(:,i),i)=servePeople;
end
servePeople_num=sum(serve_people,2); %投放点服务人数
%删除不服务于任何小区的投放点
delp=Delivery_num==0;
DeliveryPoint(delp,:)=[]; %投放点坐标
Delivery_num(delp)=[];  %服务小区数
DeliveryPointServe_R(delp)=[];  %服务半径
servePeople_num(delp)=[];  %服务人数
DeliveryVegWeight=servePeople_num*0.4/1000;  %投放点蔬菜存放量（吨）
serveCommunity(delp,:)=[];
%%
%确定物资聚集点
S_total=S_total(point(:,1),DeliveryPoint(:,1));
SPN=repmat(servePeople_num',length(bestindividual),1);
work=S_total.*SPN;
[~,ind]=min(sum(work,2));
rallyPoint=point(ind,:); %聚集点
%%
%输出物资聚集点到投放点的最短路径
global shortestPath;
global pointer;
DeliveryPoint_num=size(DeliveryPoint,1);
pathCells=cell(DeliveryPoint_num,1);
pathRow=zeros(DeliveryPoint_num,1);
for i=1:DeliveryPoint_num
    shortestPath=zeros(9932,2);
    pointer=1;
    FloydPathOutput(P,rallyPoint(1,1),DeliveryPoint(i,1));
    pathRow(i)=sum(shortestPath(:,1)~=0);
    pathCells{i}=shortestPath(1:pathRow(i),:);
end

Ral2DelPath=zeros(sum(pathRow),3); %物资聚集点到投放点的最短路径
Start=1;
for i=1:DeliveryPoint_num
    Ral2DelPath(Start:Start+pathRow(i)-1,1)=DeliveryPoint(i,1);
    Ral2DelPath(Start:Start+pathRow(i)-1,2:3)=pathCells{i};
    Start=Start+pathRow(i);
end
%%
%输出投放点到小区的最短路径
deliveryPathCells=cell(DeliveryPoint_num,1);
deliveryPathRow=zeros(DeliveryPoint_num,1);
for i=1:DeliveryPoint_num
    serveComP=community(serveCommunity(i,:),[1 7]);
    comPathCells=cell(Delivery_num(i),1);
    comPathRow=zeros(Delivery_num(i),1);
    for j=1:Delivery_num(i)
        shortestPath=zeros(9932,2);
        pointer=1;
        FloydPathOutput(P,DeliveryPoint(i,1),serveComP(j,2));
        comPathRow(j)=sum(shortestPath(:,1)~=0);
        comPathCells{j}=shortestPath(1:comPathRow(j),:);
    end
    Del2ComPath=zeros(sum(comPathRow),3); %投放点到小区的最短路径
    Start=1;
    for j=1:Delivery_num(i)
        Del2ComPath(Start:Start+comPathRow(j)-1,1)=serveComP(j,1);
        Del2ComPath(Start:Start+comPathRow(j)-1,2:3)=comPathCells{j};
        Start=Start+comPathRow(j);
    end
    deliveryPathCells{i}=Del2ComPath;
    deliveryPathRow(i)=Start-1;
end

allDel2ComPath=zeros(sum(deliveryPathRow),4); %所有投放点到小区的最短路径
Start=1;
for i=1:DeliveryPoint_num
    allDel2ComPath(Start:Start+deliveryPathRow(i)-1,1)=DeliveryPoint(i,1);
    allDel2ComPath(Start:Start+deliveryPathRow(i)-1,2:4)=deliveryPathCells{i};
    Start=Start+deliveryPathRow(i);
end

save('changchunallDel2ComPath','allDel2ComPath')