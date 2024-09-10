clear
name = {'COIL20','DIGIT10'};
addpath('./Data/')
rho = linspace(0.01,0.5,6)*10^3;
Acc = zeros(1,length(rho));
Nmi = zeros(1,length(rho));

repk = 20;
improve = zeros(1, 2*length(rho));
for i = 2

    Data = load(name{i});
    X = Data.fea;
    %X = X(1:1000,:);
    lab = Data.gnd;
    %lab = lab(1:1000,:);rho = [1,10,100rho = [1,10,100,1000,10000,100000];,1000,10000,100000];
    
    
    NX = diag(1./sqrt(sum(X.^2,2)))*X;
    n = length(lab);
    
    
    s = sum(X.^2,2);
    D = s*ones(1,n)+ones(n,1)*s'-2*X*X';
    ave = sum(D(:))/(n^2);
    M = exp(-D/ave/2);
    
    K = length(unique(lab));
    result1 = ADMM_SD1(M, n, K, 5);
    [U1,~] = principal_k(result1.X, K);
    [Acc, Nmi] = rep_kmeans(U1, K, lab, repk);

    for k = 1:length(rho)
        Temp0 = ADMMnm(M, rho(k), K, 0, (K/n), result1.X);
        [zz0,~] = principal_k(Temp0.X, K);
        [improve(k), improve(k+length(rho))] = rep_kmeans(zz0, K, lab, repk);
    end
    fprintf('Data:%s,Acc=%f, Nmi=%f\n',name{i}, Acc, Nmi)
    format_convert(improve);
    fprintf('\n');
end

function [U,P] = principal_k(A, k)
    [U, D]= eig(A);
    [~, ind] = sort(diag(D),'descend');
    U = U(:,ind(1:k));
    P = U*U';
end



function [Acc, Nmi] = rep_kmeans(F, class, lab, repk)
    F = diag(1./(sqrt(sum(F.^2,2))+eps))*F;
    lab = round(lab);
    acc = zeros(1,repk);
    nmii = zeros(1,repk);
    for i = 1:repk
        idx = round(kmeans(F, class));
        [acc(i),~] = calAC(idx',lab');
        nmii(i) = nmi(lab', idx');
    end
    Acc = mean(acc);
    Nmi = mean(nmii);
end

