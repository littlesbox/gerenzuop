%图5-11 新增感染人数-供应方案离散图程序
clear;
data=xlsread('蔬菜包.xlsx'); 
sample_num=size(data,1);
date=linspace(datenum(2022,3,29),datenum(2022,5,1),34);
chaoyang=data(:,2);
jingkai=data(:,3);
changcun=data(:,4);
nanguan=data(:,5);
erdao=data(:,6);
jingyue=data(:,7);
qikai=data(:,8);
lvyuan=data(:,9);
kuancheng=data(:,10);
newAdd=data(:,11);

fig=figure;
left_color = [0 0 0];
right_color = [0 0 0];
set(fig,'defaultAxesColorOrder',[left_color; right_color]);

yyaxis left;
plot(date,chaoyang,'b.','MarkerSize',12);
hold on;
plot(date,jingkai,'g.','MarkerSize',12);
plot(date,changcun,'.','color',[0.1,0.6,0.8],'MarkerSize',12);
plot(date,nanguan,'c.','MarkerSize',12);
plot(date,erdao,'m.','MarkerSize',12);
plot(date,jingyue,'y.','MarkerSize',12);
plot(date,qikai,'k.','MarkerSize',12);
plot(date,lvyuan,'.','color',[0.5,0.7,0],'MarkerSize',12);
plot(date,kuancheng,'.','color',[0,0.8,0.5],'MarkerSize',12);
ylabel('蔬菜发放量（吨）');

yyaxis right;
plot(date,newAdd,'r^','LineWidth',2);
ylabel('新增确诊人数');
legend('朝阳区','经开区','长春新区','南关区','二道区','净月区','汽开区','绿园区','宽城区','新增确诊');
dateaxis('x',6);
xlabel('日期');
