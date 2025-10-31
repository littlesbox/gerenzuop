load('road.mat');
load('center.mat')

for i=1:length(roaden)
   a=roadsta(i);
   b=roaden(i);
   x=[roadloc_x(a);roadloc_x(b)];
   y=[roadloc_y(a);roadloc_y(b)];
   plot(x,y,'Color',[0.5333 0.8117 0.9216]);
   hold on;
end

si=10;
plot(center_x,center_y,'*','MarkerSize',si,'linewidth',2)