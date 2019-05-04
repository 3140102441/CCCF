clear
%load rating   %the raw data

%generate a toy random user-item matrix of 100 users and 50 items.
Train = rand(100,50);
Train(Train>0.5) = 0;
Train = Train*5;
Train = ceil(Train);
Train = sparse(Train);

%no-zero index
IDX = (Train~=0);
IDXt = (Train'~=0);

[user_cnt,item_cnt] = size(Train);

%parameter setting
max_value = 5;
min_value = 1;
alpha=1e-3;
beta=1e-3;
isinit = false;
maxItr =10;
tol = 1e-7;
ModelNum = 8;
r = 8;
width = 0.7;


%pick anchor point and calculate weight
[totalu,totalv] = cal_weight(Train,r,ModelNum,user_cnt,item_cnt,width);

%Train
train;






