%计算个体的适应值
%Name:calfitvalue.m

function fitvalue=calfitvalue(pop)
%fitvalue：适应度
%pop：种群
global community;
global D;
global S;
[px,~]=size(pop);
[community_num,~]=size(community);
fitvalue=zeros(px,1);
serve_R=zeros(community_num,1); %服务半径
%计算种群适应度值
for i=1:px
    %%
    %确定服务半径
    for j=1:community_num
        serve_R(j)=min(D(logical(pop(i,:)),j));
    end
    currentServeR=max(serve_R);
    %%
    %确定服务小区
    pxD=D(logical(pop(i,:)),:);
    serveCommunity=pxD<=currentServeR;
    %%
    %确定工作量
    pxS=S(logical(pop(i,:)),:);
    workload=0;
    for j=1:community_num
        distance=pxS(serveCommunity(:,j),j);
        distanceTotal=sum(distance);
        weight=distance/distanceTotal;
        [~,ind]=sort(distance);
        descWeight=sort(weight,'descend');
        weight(ind)=descWeight; %按照距离分配权重，距离越近，权重越大
        servePeople=weight*community(j,4);
        workload=workload+(servePeople')*distance;
    end
    fitvalue(i)=100/log(10+workload);
end
end
