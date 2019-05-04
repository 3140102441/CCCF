function [ U,V ] = MF(Train,code_len)
%MF by SGD
[N,M] = size(Train);
k = code_len;
P = rand(N,k);
Q = rand(M,k);
[U,V] = matrix_factorization(Train,P,Q);


end

function [U,V] = matrix_factorization(R,P,Q)
Q = Q';
steps=10;
alpha=0.001;
beta=0.01;
[N,~]=size(R);
for step = 1:steps
    for i = 1:N
        [~, j ,~]=find(R(i,:));
        for x = 1:size(j,2)
            eij = R(i,j(x)) - P(i,:)*Q(:,j(x));
            
            P(i,:) = P(i,:) + alpha * (2 * eij * Q(:,j(x))' - beta * P(i,:));
            Q(:,j(x)) = Q(:,j(x)) + alpha * (2 * eij * P(i,:)' - beta * Q(:,j(x)));
            
        end
    end
end
U = P;
V = Q';
end


