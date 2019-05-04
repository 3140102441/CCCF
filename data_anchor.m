function [ AnchorPointU,AnchorPointV ] = data_anchor( U_init,V_init,ModelNum)
%pick anchorpoint by cluster
[~,~,~,~,midx] = kmedoids(U_init,ModelNum);
AnchorPointU = midx;
[~,~,~,~,midx] = kmedoids(V_init,ModelNum);
AnchorPointV = midx;
end




