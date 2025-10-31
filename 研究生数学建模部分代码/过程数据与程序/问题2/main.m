clear
clc
%%
%筛选区域
global point;
global community;
global D;
global S;
%数字对应城区，例如“1”对应的是宽城区，具体对应在“附件3：长春市9个区交通网络数据和主要小区相关数据”中
[point,community,D,S]=citySelect(3); 
%% 
popsize=100;                             	
%种群大小
chromlength=length(point);                     		
%字符串长度(个体长度)
pc=0.6;                                    	
%交叉概率，只有在随机数小于pc时，才会产生交叉
pm=0.001;                               	
%变异概率
pop=initpop(popsize,chromlength);	
%随机产生初始群体
for i=1:100                                                           	
%100为遗传代数
		
        fitvalue=calfitvalue(pop);
        %计算种群中个体目标函数值和适应度值
		
        [newpop]=selection(pop,fitvalue);
		%复制    
        [newpop1]=crossover(newpop,pc); 
		%交叉
        [newpop2]=mutation(newpop1,pc);
		%变异   

        fitvalue=calfitvalue(newpop2);              
		%计算种群中个体目标函数值和适应度值 

        [bestindividual,bestfit]=best(newpop2,fitvalue);
		%求出群体中适应度值最大的个体及其适应度值
        
        fitvaluemean=mean(fitvalue);
        y1=fitvaluemean;
        y2=bestfit;
 
        plot(i,y1,'r*',i,y2,'g*');
        hold on
        %title('投放点选址问题遗传算法求解');
        xlabel('进化代数');
        ylabel('目标值');
        legend('平均适应度','最大适应度','location','best');
        pop=newpop2;
        %更新种群
end

[z,index]=max(fitvalue);
bestprice=z;
bestscheme=bestindividual;

save chaoyang
