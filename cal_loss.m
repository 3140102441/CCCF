function [result] = cal_loss(Test,predict,IDXt)
%calculate mse
[user_cnt,~] = size(Test);
result = 0;
for i = 1 :user_cnt 
    loss = Test(i,IDXt(:,i)) - predict(i,IDXt(:,i));
    result = result + loss*loss';
end
end

