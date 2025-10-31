clear;
%Floyd算法计算路口间的最短距离
load('path.mat');
[S,P]=FloydSPR(path);
save SPR