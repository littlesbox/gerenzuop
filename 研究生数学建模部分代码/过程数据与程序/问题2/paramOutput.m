clear;
load('chaoyang.mat');
DeliveryPoint=point(logical(bestindividual),2:3);
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

save('chaoyangDeliverPoint','DeliveryPoint','Delivery_num','DeliveryPointServe_R','servePeople_num')
