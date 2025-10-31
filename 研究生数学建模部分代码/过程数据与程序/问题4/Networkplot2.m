%clear
%clc
% load('house.mat');
% load('housetypeindex.mat');
load('changchunRal2DelPath');
load('changchunallDel2ComPath.mat');
load('changchunDeliverPoint.mat');
load('road.mat');
load('house.mat');
load('housetypeindex.mat');

A=Ral2DelPath;
B=allDel2ComPath;

rdsta=[A(:,end-1);B(:,end-1)];
rden=[A(:,end);B(:,end)];
for i=1:length(rden)
   a=rdsta(i);
   b=rden(i);
   x=[roadloc_x(a);roadloc_x(b)];
   y=[roadloc_y(a);roadloc_y(b)];
   plot(x,y,'Color',[0.5333 0.8117 0.9216]);
   hold on;
end
s=15;
si=3;
plot(DeliveryPoint(:,1),DeliveryPoint(:,2),'.','MarkerSize',s)
plot(roadloc_x(rdsta(1)),roadloc_y(rdsta(1)),'*','MarkerSize',10,'linewidth',2)
% legend('路径','投放点','聚集中心')
% plot(roadloc_x(3148),roadloc_y(3148))
% si=15;
%plot(houseloc_x(kuanInd),houseloc_y(kuanInd),'o','MarkerSize',si)
%plot(houseloc_x(erdaoInd),houseloc_y(erdaoInd),'o','MarkerSize',si)
%plot(houseloc_x(chaoInd),houseloc_y(chaoInd),'o','MarkerSize',si)
%plot(houseloc_x(lvInd),houseloc_y(lvInd),'o','MarkerSize',si)
%plot(houseloc_x(nanguanInd),houseloc_y(nanguanInd),'o','MarkerSize',si)
%plot(houseloc_x(jingkaiInd),houseloc_y(jingkaiInd),'o','MarkerSize',si)
plot(houseloc_x(changInd),houseloc_y(changInd),'o','MarkerSize',si)
%plot(houseloc_x(jingyueInd),houseloc_y(jingyueInd),'o','MarkerSize',si)
%plot(houseloc_x(qiInd),houseloc_y(qiInd),'o','MarkerSize',si)
% axis equal