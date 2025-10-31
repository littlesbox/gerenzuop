%clear
%clc
load('house.mat');
load('housetypeindex.mat');
load('road.mat');
for i=1:length(roaden)
   a=roadsta(i);
   b=roaden(i);
   x=[roadloc_x(a);roadloc_x(b)];
   y=[roadloc_y(a);roadloc_y(b)];
   plot(x,y,'Color',[0.5333 0.8117 0.9216]);
   hold on;
end
si=15;
plot(houseloc_x(kuanInd),houseloc_y(kuanInd),'*','MarkerSize',si)
plot(houseloc_x(erdaoInd),houseloc_y(erdaoInd),'*','MarkerSize',si)
plot(houseloc_x(chaoInd),houseloc_y(chaoInd),'*','MarkerSize',si)
plot(houseloc_x(lvInd),houseloc_y(lvInd),'*','MarkerSize',si)
plot(houseloc_x(nanguanInd),houseloc_y(nanguanInd),'*','MarkerSize',si)
plot(houseloc_x(jingkaiInd),houseloc_y(jingkaiInd),'*','MarkerSize',si)
plot(houseloc_x(changInd),houseloc_y(changInd),'*','MarkerSize',si)
plot(houseloc_x(jingyueInd),houseloc_y(jingyueInd),'*','MarkerSize',si)
plot(houseloc_x(qiInd),houseloc_y(qiInd),'*','MarkerSize',si)
axis equal