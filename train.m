total_modelB = cell(1,ModelNum);
total_modelD = cell(1,ModelNum);
predict = sparse(user_cnt,item_cnt);
[m,n] = size(Train);
converge = false;
currErr = Inf;
it = 1;

Train = ScaleScore(Train,r, max_value,min_value,ModelNum);
%%init
if isinit
    [total_modelU,total_modelV,total_modelX,total_modelY]=CCCF_init(Train,ModelNum,r,user_cnt,item_cnt,totalu,totalv,IDX,IDXt);
else
    total_modelU = cell(1,ModelNum);
    total_modelV = cell(1,ModelNum);
    total_modelX = cell(1,ModelNum);
    total_modelY = cell(1,ModelNum);
    parfor modelcnt = 1:ModelNum
        total_modelU{1,modelcnt} = randn(r,user_cnt);
        total_modelV{1,modelcnt} = randn(r,item_cnt);
        total_modelX{1,modelcnt} = UpdateSVD(total_modelU{1,modelcnt});
        total_modelY{1,modelcnt} = UpdateSVD(total_modelV{1,modelcnt});
    end
end

for i = 1: ModelNum
    total_modelB{1,i} = sign(total_modelU{1,i});
    total_modelD{1,i} = sign(total_modelV{1,i});
    total_modelB{1,i}(total_modelB{1,i} == 0) = 1;
    total_modelD{1,i}(total_modelD{1,i} == 0) = 1;
    predict = predict + total_modelB{1,i}'*total_modelD{1,i}.*(totalu(:,i) * totalv(:,i)');
end

while ~converge
    %update model
    for  modelcnt = 1:ModelNum
        B  = total_modelB{1,modelcnt};
        D =  total_modelD{1,modelcnt};
        totalwij = totalu(:,modelcnt) * totalv(:,modelcnt)';
        newS = Train - predict  + B'*D.*totalwij;
        %update B
        for i = 1:m
            b = B(:,i);
            d = repmat(totalwij(i,IDXt(:,i)),r,1).*D(:,IDXt(:,i));
            DCDmex(b,d*d',d*newS(i,IDXt(:,i))',alpha*total_modelX{1,modelcnt}(:,i),maxItr);
            B(:,i) = b;
        end
        %update D
        for j = 1:n
            b = repmat(totalwij(IDX(:,j),j)',r,1).*B(:,IDX(:,j));
            d = D(:,j);
            DCDmex(d,b*b',b*newS(IDX(:,j),j), beta*total_modelY{1,modelcnt}(:,j),maxItr);
            D(:,j)= d;
        end
        predict = predict+(B'*D-total_modelB{1,modelcnt}'*total_modelD{1,modelcnt}).*totalwij;
        total_modelB{1,modelcnt} = B;
        total_modelD{1,modelcnt} = D;
        %update X
        total_modelX{1,modelcnt} = UpdateSVD(B);
        %update Y
        total_modelY{1,modelcnt} = UpdateSVD(D);
    end
    prevErr = currErr;
    %calculate loss
    currErr = cal_loss(Train,predict,IDXt);
    for modelcnt = 1:ModelNum
        currErr  = currErr -2*alpha*trace(total_modelB{1,modelcnt}*total_modelX{1,modelcnt}')-2*beta*trace(total_modelD{1,modelcnt}*total_modelY{1,modelcnt}');
    end
    if (it >= maxItr || (prevErr - currErr) < max([user_cnt,item_cnt])*tol)
        converge = true;
    end
    
    it = it+1;
    
end




