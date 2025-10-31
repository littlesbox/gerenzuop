clear
clc
load new304514
t=1:180;
t=t';

N=9066906;
S=zeros(length(t),1);
E=zeros(length(t),1);
I=zeros(length(t),1);
R=zeros(length(t),1);
a=0.6;
p=3;
b=0.125;
c=0.6;
I(1)=5;
S(1)=N-I(1);
E(1)=1;
R(1)=1;
inew=zeros(length(t),1);
for i=1:length(t)-1
    S(i+1)=-p*a*S(i)*I(i)/N+S(i);
    E(i+1)=p*a*S(i)*I(i)/N-b*E(i)+E(i);
    I(i+1)=b*E(i)-c*I(i)+I(i);
    R(i+1)=c*I(i)+R(i);
    inew(i+1)=I(i+1)-I(i);
end

plot(t(1:72),inew(1:72),t(1:72),new304514,[23;23],[-2000,12000],'k')
legend('预测新增','实际新增','开始发放蔬菜包')
figure
plot(t,S,t,E,t,I,t,R)
legend('易感者','潜伏者','感染者','移出者')


