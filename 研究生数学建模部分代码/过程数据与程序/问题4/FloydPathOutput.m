% 本函数用于输出由弗洛伊德算法得到的最短路径
% P：函数 function [S, P]=FloydSPR(AdjMax)所返回的矩阵P
% b：查询路径的起点
% e：查询路径的终点
function FloydPathOutput(P,b,e)
global shortestPath;
global pointer;
if P(b,e)==-1   % 若P(b,e)对应的值为-1，则直接输出该条路径
    shortestPath(pointer,:)=[b,e];
    pointer=pointer+1;
else   % 否则以P(b,e)的值分别作为终点和起点递归调用函数自身
    FloydPathOutput(P,b,P(b,e));
    FloydPathOutput(P,P(b,e),e);
end
end