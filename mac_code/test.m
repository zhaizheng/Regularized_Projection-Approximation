%%
addpath('../')
%[S, real_A, lab] = data_generation(0.35, 0.6);
%%
k = 2;
%save 'data.mat' S real_A lab
load("data.mat")


%%




figure
tiledlayout(3,6,"TileSpacing","compact")

%%
Lambda = [10,100,1000,1000,10000];
Str = {'10^1','10^2','10^3','10^4','10^5'};
a = 0;
b = 1/20;
Lip = 400;%[50,100,200,400,800,1600,3200];
class = 2;
repk = 20;
result_n = zeros(1,length(Lambda));
Acc_n = zeros(1,length(Lambda));
Nmi_n = zeros(1,length(Lambda));
nexttile
imagesc(S);
title('Original','FontSize',16)
axis off
% figure
% tiledlayout(1,6,"TileSpacing","compact")
for i = 1:length(Lambda)
        rez = ADMMnm(S, Lambda(i), k, a, b, S);
        A = rez.X;
        result_n(i) = norm(A-real_A/20,'fro');
        [U,P] = principal_k(A, class);
        [Acc_n(i), Nmi_n(i)] = rep_kmeans(U, class, lab, repk);
        nexttile
        imagesc(rez.X)
        title(['Bounded ','\lambda=', Str{i}],'FontSize',16);
        axis off
end
fprintf('\n')
format_print([result_n, Acc_n, Nmi_n])
%%
%%[acc_n, nmi_n] = rep_kmeans(rez.initU, class, lab, repk);
%%

%%
Lambda = [10,100,1000,1000,10000];
Lip = 2;%[50,100,200,400,800,1600,3200];
class = 2;
repk = 20;
result_n = zeros(1,length(Lambda));
Acc_n = zeros(1,length(Lambda));
Nmi_n = zeros(1,length(Lambda));

nexttile
imagesc(S);
title('Original','FontSize',16)
axis off

% figure
% tiledlayout(1,6,"TileSpacing","compact")
for i = 1:length(Lambda)
        rez = ADMM_positive(S, Lambda(i), k, Lip);
        A = rez.X;
        result_n(i) = norm(A-real_A/20,'fro');
        [U,P] = principal_k(A, class);
        [Acc_n(i), Nmi_n(i)] = rep_kmeans(U, class, lab, repk);
        nexttile
        imagesc(rez.X)
        title(['Positive ','\lambda=',Str{i}],'FontSize',16);
        axis off
end
fprintf('\n')
format_print([result_n, Acc_n, Nmi_n])
%%
%%[acc_n, nmi_n] = rep_kmeans(rez.initU, class, lab, repk);

Lambda = 0.1:0.1:0.5;
Lip = 40;%[50,100,200,400,800,1600,3200];
Delta = [0.001,0.0001,0.00001,0.000001,0.0000001];
class = 2;
repk = 20;
result = zeros(length(Lambda),length(Delta));
Acc = zeros(length(Lambda),length(Delta));
Nmi = zeros(length(Lambda),length(Delta));
for i = 1:length(Lambda)
    for j = 1:length(Delta)
        rez = ADMM_sparse(S, Lambda(i), k, Delta(j), Lip);
        A = truncate(rez.X, Delta(j));
        result(i,j) = norm(A-real_A/20,'fro');
        [U,P] = principal_k(A, class);
        [Acc(i,j), Nmi(i,j)] = rep_kmeans(U, class, lab, repk);
    end
end
format_print([result', Acc', Nmi'])
%%
[acc, nmi] = rep_kmeans(rez.initU, class, lab, repk);
ZU = rez.initU*rez.initU';
fprintf('original acc:%f, nmi:%f,dist=:%f\n',acc,nmi, norm(ZU-real_A/20,'fro'));
%%
figure
tiledlayout(1,6,"TileSpacing","compact")
nexttile
imagesc(S);
title('Original','FontSize',16)
axis off
re = cell(1,5);
delta = 0.000001; lambda = 0.1; k = 2; lip=40;
Lambda = 0.1:0.4:1.7;
for i = 1:5
    re{i} = ADMM_sparse(S, Lambda(i), k, delta, lip);
    nexttile
    imagesc(re{i}.X)
    title(['Sparse ','\lambda=', num2str(Lambda(i))],'FontSize',16);
    axis off
end




function A = truncate(A, mu)
    A(A<mu) = 0;
end


function [U,P] = principal_k(A, k)
    [U, ~]= eigs(A,k);
    P = U*U';
end



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