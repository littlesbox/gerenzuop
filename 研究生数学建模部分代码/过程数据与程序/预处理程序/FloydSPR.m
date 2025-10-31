function [S, P]=FloydSPR(AdjMax)
% *INPUT:* 
% AdjMax: Adjacent matrix that represents a weighted, directed graph
% 若两个点之间不可达，路径长度为Inf
% *OUTPUT:*
% S: distance to destination node
% P: next hop node
%
% *DESCRIPTION*
% Given a input adjacent matrix (AdjMax) that represents a weighted, directed graph. 
% The function finds the shorest path from one vertex 'i' to another 'j'. 
% The return values includes a matrix (S) that denotes the shortest distance 
% between vertices 'i' and 'j', and a matrix (P) that denotes the next vertex 'k' 
% on the path from vertex 'i' to vertex 'j' 

N=min(length(AdjMax(:,1)),length(AdjMax(1,:)));
% P是Path矩阵存储中途点矩阵
P=-1*ones(N,N);
% S储存两点间的最短路径长度
S=AdjMax;
% 以点k作为中间点，循环更新点i到点j的最短路径
for k=1:N
    for i=1:N
        for j=1:N
            if S(i,k)==inf
                continue;
            end
            if S(k,j)==inf
                continue;
            end
            % S(i,j)是当前i和j两点间的最短路径长度
            % S(i,k)+S(k,j)是以点k作为中间点后的路径长度
            if S(i,j)>S(i,k)+S(k,j)
                % 如果i和k之间没有中间点，就将k作为i和j之间的中间点
                if P(i,k)==-1
                    P(i,j)=k;   
                % 否则，把i和k之间的中间点作为i和j之间的中间点
                % （即从i到j最短路径的下一个点）
                else
                    P(i,j)=P(i,k);
                end
                S(i,j)=S(i,k)+S(k,j);
            end
        end
    end
end

end
