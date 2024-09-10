addpath('./Data/')
name = {'COIL10','COIL20','DIGIT5','DIGIT10'};

Data = load('COIL10');
X = Data.fea;
lab = round(Data.gnd);

NX = diag(1./sqrt(sum(X.^2,2)))*X;
n = length(lab);


s = sum(X.^2,2);
D = s*ones(1,n)+ones(n,1)*s'-2*X*X';
ave = sum(D(:))/(n^2);
M = exp(-D/ave/2);

K = length(unique(lab));
class = K;  repk = 50;
%%
Lambda = 0.1:0.1:0.8;
Lip = 40;
Delta = [0.001,0.0001,0.00001,0.000001,0.0000001];


Acc_sparse = zeros(length(Lambda),length(Delta));
Nmi_sparse = zeros(length(Lambda),length(Delta));
for k = 1:length(Lambda)
    for s = 1:length(Delta)
        rez = ADMM_sparse(M, Lambda(k), class, Delta(s), Lip, M);
        B = truncate(rez.X, Delta(s));
        %result(i,j) = norm(A-real_A/20,'fro');
        [U,P] = principal_k(B, class);
        [Acc_sparse(k,s), Nmi_sparse(k,s)] = rep_kmeans(U, class, lab, repk);
        fprintf('RBP clustering k=%d,s=%d, acc=%f,nmi=%f\n',k, s, max(Acc_sparse(:)),max(Nmi_sparse(:)));
    end
end
figure

bar3(Acc_sparse);

%%









function [U,P] = principal_k(A, k)
    [U, D]= eig(A);
    [~, ind] = sort(diag(D),'descend');
    U = U(:,ind(1:k));
    P = U*U';
end

