function [totalu,totalv] = cal_weight(Train,r,ModelNum,user_cnt,item_cnt,width)
%u-weight
totalu = zeros(user_cnt,ModelNum);
%v-weight
totalv = zeros(item_cnt,ModelNum);

%tradictional MF, you can also load the pretrained U,V matrix
[U_init,V_init] = MF(Train,r);
%pick anchor point by cluster
[AnchorPointU,AnchorPointV] = data_anchor(U_init,V_init,ModelNum);

U_init_row_norm = sum(abs(U_init).^2,2).^(1/2);
V_init_row_norm = sum(abs(V_init).^2,2).^(1/2);

%calculate weight
parfor modelcnt = 1:ModelNum
    anchoru = AnchorPointU(modelcnt);
    anchorv = AnchorPointV(modelcnt);
    
    simu = 1-2.0/pi*acos(U_init*U_init(anchoru,:)'./((U_init_row_norm).*(repmat(norm(U_init(anchoru,:)),user_cnt,1))));
    
    simu(isnan(simu)) = 0;
    simuu = 0.75*(1-((1-simu)/width).^2);
    simuu(simuu<0) = 0;
    totalu(:,modelcnt) = simuu;
    
    simv = 1-2.0/pi*acos(V_init*V_init(anchorv,:)'./((V_init_row_norm).*(repmat(norm(V_init(anchorv,:)),item_cnt,1))));
    
    simv(isnan(simv)) = 0;
    simvv = 0.75*(1-((1-simv)/width).^2);
    simvv(simvv<0) = 0;
    totalv(:,modelcnt) = simvv;
end
totalu = totalu./(repmat(sum(totalu,2),1,ModelNum));
totalu(isnan(totalu)) = 0;
totalv = totalv./(repmat(sum(totalv,2),1,ModelNum));
totalv(isnan(totalv)) = 0;
end