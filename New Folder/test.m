%%
addpath('../')
[S, real_A, lab] = data_generation(0.4, 0.6);
%%
tiledlayout(1,5,"TileSpacing","compact")
nexttile
imagesc(S);
re = cell(1,5);

delta = 0.000001; lambda = 0.25; k = 2; lip=200;
re{1} = ADMM_sparse(S, lambda, k, delta, lip);
nexttile
imagesc(re{1}.X)

nexttile
lambda = 0.5;
re{2} = ADMM_sparse(S, lambda, k, delta, lip);
imagesc(re{2}.X)

nexttile
lambda = 1;
re{3} = ADMM_sparse(S, lambda, k, delta, lip);
imagesc(re{3}.X)

nexttile
lambda = 2;
re{4} = ADMM_sparse(S, lambda, k, delta, lip);
imagesc(re{4}.X)


%%

Lambda = [0.125,0.25,0.5,1,2,4];
Lip = 400;%[50,100,200,400,800,1600,3200];
Delta = [0.1,0.01,0.001,0.0001,0.00001,0.000001,0.0000001];
class = 2;
repk = 20;
result = zeros(length(Lambda),length(lip));
for i = 1:length(Lambda)
    for j = 1:length(Delta)
        rez = ADMM_sparse(S, Lambda(i), k, Delta(j), Lip);
        result(i,j) = norm(rez.X-real_A/20,'fro');
        [Acc(i,j), Nmi(i,j)] = rep_kmeans(rez.U, class, lab, repk);
    end
end
%%
[acc, nmi] = rep_kmeans(rez.initU, class, lab, repk);

%%
tiledlayout(1,6,"TileSpacing","compact")
nexttile
imagesc(S);
re = cell(1,5);

delta = Delta(2); lambda = 1; k = 2; lip=200;
re{1} = ADMM_sparse(S, lambda, k, delta, lip);
nexttile
imagesc(re{1}.X)

nexttile
delta = Delta(3); 
re{2} = ADMM_sparse(S, lambda, k, delta, lip);
imagesc(re{2}.X)

nexttile
delta = Delta(4); 
re{3} = ADMM_sparse(S, lambda, k, delta, lip);
imagesc(re{3}.X)

nexttile
delta = Delta(5); 
re{4} = ADMM_sparse(S, lambda, k, delta, lip);
imagesc(re{4}.X)

nexttile
delta = Delta(6); 
re{5} = ADMM_sparse(S, lambda, k, delta, lip);
imagesc(re{5}.X)


function [S, A, label] = data_generation(signal, noise)
    k = 2;
    each_k = 20;
    n = k*each_k;
    S =  zeros(n,n);
    label = [];
    for j = 1:k
 %       S((j-1)*each_k+1:j*each_k,(j-1)*each_k+1:j*each_k) = 1;
        SIGNAL = rand(n/2,n/2) > signal;
        SIGNAL_Temp1 = triu(SIGNAL,1);
        SIGNAL_Temp2 = triu(SIGNAL);
        S((j-1)*each_k+1:j*each_k,(j-1)*each_k+1:j*each_k) = SIGNAL_Temp2 +SIGNAL_Temp1';
        label = [label, j*ones(1,each_k)];
    end
    E =  rand(n/2,n/2)> noise;
    E2 = [zeros(n/2),E;E',zeros(n/2)];
    A = [ones(n/2),zeros(n/2);
        zeros(n/2),ones(n/2)];
    S = S+E2;
end