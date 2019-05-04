function [total_modelU,total_modelV,total_modelX,total_modelY] = CCCF_init(Train,ModelNum,r,user_cnt,item_cnt,totalu,totalv,IDX,IDXt)
total_modelU = cell(1,ModelNum);
total_modelV = cell(1,ModelNum);
total_modelX = cell(1,ModelNum);
total_modelY = cell(1,ModelNum);
predict = 0;
for modelcnt = 1:ModelNum
    total_modelU{1,modelcnt} = randn(r,user_cnt);
    total_modelV{1,modelcnt} = randn(r,item_cnt);
    total_modelX{1,modelcnt} = UpdateSVD(total_modelU{1,modelcnt});
    total_modelY{1,modelcnt} = UpdateSVD(total_modelV{1,modelcnt});
    UVmatrix = total_modelU{1,modelcnt}'*total_modelV{1,modelcnt};
    weight = totalu(:,modelcnt)*totalv(:,modelcnt)';
    predict = predict + UVmatrix.*weight;
end

%parameter setting 
alpha = 1e-3;
beta = 1e-3;
converge = false;

currErr = Inf;
it = 1;
maxItr = 10;
tol = 1e-7;

while ~converge
    %update model
    for  modelcnt = 1:ModelNum
        U  = total_modelU{1,modelcnt};
        V =  total_modelV{1,modelcnt};
        totalwij = totalu(:,modelcnt) * totalv(:,modelcnt)';
        newS = Train - predict + U'*V.*totalwij;
        %update U
        for i = 1:user_cnt
            Vi = repmat(totalwij(i,IDXt(:,i)),r,1).*V(:,IDXt(:,i));
            temps = newS(i,IDXt(:,i))';
            if isempty(temps)
                continue;
            end
            Q = Vi*Vi'+alpha*eye(r);
            L = Vi*temps+2*alpha*total_modelX{1,modelcnt}(:,i);
            U(:,i) = Q\L;
        end
        %update V
        for j = 1:item_cnt
            Uj = repmat(totalwij(IDX(:,j),j)',r,1).*U(:,IDX(:,j));
            temps = newS(IDX(:,j),j);
            if isempty(temps)
                continue;
            end
            Q = Uj*Uj'+beta*eye(r);
            L = Uj*temps+2*beta*total_modelY{1,modelcnt}(:,j);
            V(:,j) = Q\L;
        end
        predict = predict+(U'*V-total_modelU{1,modelcnt}'*total_modelV{1,modelcnt}).*totalwij;
        total_modelU{1,modelcnt} = U;
        total_modelV{1,modelcnt} = V;
        %update X
        total_modelX{1,modelcnt} = UpdateSVD(U);
        %update Y
        total_modelY{1,modelcnt} = UpdateSVD(V);
    end
    prevErr = currErr;
    %calculate loss
    currErr = cal_loss(Train,predict,IDXt);
    for modelcnt = 1:ModelNum
        currErr  = currErr -2*alpha*trace(total_modelU{1,modelcnt}*total_modelX{1,modelcnt}')-2*beta*trace(total_modelV{1,modelcnt}*total_modelY{1,modelcnt}');
        normu = norm(total_modelU{1,modelcnt});
        normv = norm(total_modelV{1,modelcnt});
        currErr = currErr + alpha*normu.^2+beta*normv.^2;
    end
    if (it >= maxItr || (prevErr - currErr) < max([user_cnt,item_cnt])*tol)
        converge = true;
    end
    it = it+1;
    
end
end