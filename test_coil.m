addpath('./Data/')
name = {'COIL10','COIL20','DIGIT5','DIGIT10'};
ind = [1,4];
tiledlayout(2,6,'TileSpacing','Compact')
for t = 1:length(ind)
    i = ind(t);
Data = load(name{i});
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

Lambda = 0.1:0.1:0.8;
Lip = 40;
Delta = [0.001,0.0001,0.00001,0.000001,0.0000001];
% class = 2;
% repk = 20;
% result = zeros(length(Lambda),length(Delta));
%Lambda=0.3; Delta=0.1^5;



result1 = ADMM_SD1(M, n, K, 5);
[U1,~] = principal_k(result1.X, K);
[Acc_SDP, Nmi_SDP] = rep_kmeans(U1, class, lab, repk);
fprintf('SDP clustering acc=%f,nmi=%f\n',Acc_SDP,Nmi_SDP);


result2 = ADMM_SD2(M, n, K, 5);
[U2,~] = principal_k(result2.X, K);
[Acc_SDP2, Nmi_SDP2] = rep_kmeans(U2, class, lab, repk);
fprintf('SDP2 clustering acc=%f,nmi=%f\n',Acc_SDP2,Nmi_SDP2);



[U3,P3] = principal_k(M, K);
[Acc_spec, Nmi_spec] = rep_kmeans(U3, class, lab, repk);
fprintf('spectral clustering acc=%f,nmi=%f\n',Acc_spec,Nmi_spec);



fname = 'ell_F^2'; eta = floor(n^2/K);  theta = 1; tau = 0.001;
[SLSA,U] = SSL2(fname,M,eta, K, theta, tau); 
[U4,~] = principal_k(SLSA, K);
[SLSA_ACC, SLSA_NMI] = rep_kmeans(U4, class, lab, repk);
fprintf('SLSA clustering acc=%f,nmi=%f\n',SLSA_ACC,SLSA_NMI);




Acc_sparse = zeros(length(Lambda),length(Delta));
Nmi_sparse = zeros(length(Lambda),length(Delta));
for k = 4%1:length(Lambda)
    for s = 2%1:length(Delta)
        rez = ADMM_sparse(M, Lambda(k), class, Delta(s), Lip, M);
        B = truncate(rez.X, Delta(s));
        %result(i,j) = norm(A-real_A/20,'fro');
        [U,P] = principal_k(B, class);
        [Acc_sparse(k,s), Nmi_sparse(k,s)] = rep_kmeans(U, class, lab, repk);
        k
        s
    end
end
fprintf('RBP clustering acc=%f,nmi=%f\n',max(Acc_sparse(:)),max(Nmi_sparse(:)));


%%




nexttile 
imagesc(M)
title('A','FontSize',16);
axis off


nexttile
imagesc(result1.X);
title('SDP-1','FontSize',16);
axis off


nexttile
imagesc(result2.X);
title('SDP-2','FontSize',16);
axis off

nexttile
imagesc(P3)
title('Spectral','FontSize',16);
axis off

nexttile
imagesc(SLSA)
title('SLSA','FontSize',16);
axis off

nexttile
imagesc(B)
title('RPMA-Sparse','FontSize',16);
axis off


end




function [U,P] = principal_k(A, k)
    [U, D]= eig(A);
    [~, ind] = sort(diag(D),'descend');
    U = U(:,ind(1:k));
    P = U*U';
end

