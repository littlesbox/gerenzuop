%图5-12 模型拟合的新增确诊与实际新增程序
clear;
data=xlsread('蔬菜包.xlsx'); 
sample_num=size(data,1);
date=linspace(datenum(2022,3,29),datenum(2022,5,1),34);
x=[ones(sample_num,1) data(:,1:10)];

[b,bint,r,rint,stats]=regress(data(:,11),x,0.05);

Y_NiHe=b(1);
for i=1:9
    Y_NiHe=Y_NiHe+b(i+1).*data(:,i);
end

hold on;
plot(date,data(:,11),'r^','LineWidth',2);
plot(date,Y_NiHe,'r-','LineWidth',2);
ylabel('新增确诊人数');
legend('新增确诊','回归线');
dateaxis('x',6);
xlabel('日期');
R_2=1-sum((Y_NiHe-data(:,11)).^2)./sum((data(:,11)-mean(data(:,11))).^2);
str=num2str(R_2);
disp(['拟合优度R^2=',str])