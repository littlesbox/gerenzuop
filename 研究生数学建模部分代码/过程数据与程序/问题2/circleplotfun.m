function circleplotfun(x,y,r)
    alpha=0:0.01*pi:2*pi;
    for i=1:length(x)
       x1=x(i)*ones(length(alpha),1)+r(i)*cos(alpha');
       y1=y(i)*ones(length(alpha),1)+r(i)*sin(alpha');
       plot(x(i),y(i),'.','MarkerSize',20,'color',[0.5 0.5 0.5])
       hold on
       plot(x1,y1,'--','color',[0.5 0.5 0.5 0.5])
       hold on
    end
    axis equal
end