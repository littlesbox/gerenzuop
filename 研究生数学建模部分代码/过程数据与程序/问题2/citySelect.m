function [point,community,D,S]=citySelect(cityNum) 
%cityNumÎªÇøÓò±àºÅ
load('dis_piont2community.mat','community','D','point');
load('SPR.mat', 'S');
point=point(point(:,4)==cityNum,1:3);
community=community(community(:,8)==cityNum,1:7);
community(:,4)=community(:,4);
D=D(point(:,1),community(:,1));
S=S(point(:,1),community(:,7))+50;
end