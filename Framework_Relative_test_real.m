clear
name = {'COIL10','COIL20','DIGIT5','DIGIT10'};
Acc = zeros(4,5);
Nmi = zeros(4,5);
rho = linspace(0.01,0.5,6)*10^5;
Acc_bound1 = zeros(4,length(rho));
Nmi_bound1 = zeros(4,length(rho));
Acc_bound2 = zeros(4,length(rho));
Nmi_bound2 = zeros(4,length(rho));
for i = 1:4
    % if i <= 1
    %     rho = linspace(0.01,0.5,6)*10^3;
    % else
    %     rho = linspace(0.01,0.5,6)*10^5;
    % end
    addpath('./Data/')
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
    result2 = ADMM_SD2(M, n, K, 5);
    %%
    class = K;  repk = 50;
    
    [U1,~] = principal_k(result1.X, K);
    [Acc(i,1), Nmi(i,1)] = rep_kmeans(U1, class, lab, repk);
    %%
    [U2,~] = principal_k(result2.X, K);
    [Acc(i,2), Nmi(i,2)] = rep_kmeans(U2, class, lab, repk);
    
    [U3,P3] = principal_k(M, K);
    [Acc(i,3), Nmi(i,3)] = rep_kmeans(U3, class, lab, repk);
    
    
    %%
    fname = 'ell_F^2'; eta = floor(n^2/K);  theta = 1; tau = 0.001;
    [Z,U] = SSL2(fname,M,eta, K, theta, tau); 
    [U4,~] = principal_k(Z, K);
    [Acc(i,4), Nmi(i,4)] = rep_kmeans(U4, class, lab, repk);
    
    
    lambda = 0.1; n_iter = 100;
    [A, ~, ~] = CLR_zz(M, lambda, K, n_iter);
    [U5,~] = principal_k(A+A', K);
    [Acc(i,5), Nmi(i,5)] = rep_kmeans(U5, class, lab, repk);
    
    
    
    Intermedia = cell(1,5);
    Intermedia{1} = result1.X;
    Intermedia{2} = result2.X;
    Intermedia{3} = M;
    Intermedia{4} = Z;
    Intermedia{5} = A+A';


    Lambda = 0.1:0.1:0.8;
    Lip = 40;
    Delta = [0.001,0.0001,0.00001,0.000001,0.0000001];
    % class = 2;
    % repk = 20;
    % result = zeros(length(Lambda),length(Delta));
    %Lambda=0.3; Delta=0.1^5;
    Acc_sparse = zeros(length(Lambda),length(Delta));
    Nmi_sparse = zeros(length(Lambda),length(Delta));
    for k = 1:length(Lambda)
        for s = 1:length(Delta)
            rez = ADMM_sparse(M, Lambda(k), class, Delta(s), Lip, Intermedia{3});
            B = truncate(rez.X, Delta(s));
            %result(i,j) = norm(A-real_A/20,'fro');
            [U,P] = principal_k(B, class);
            [Acc_sparse(k,s), Nmi_sparse(k,s)] = rep_kmeans(U, class, lab, repk);
        end
    end
    Acc(i,6) = max(Acc_sparse(:));
    Nmi(i,6) = max(Nmi_sparse(:));
    
    %%
    
    % for k = 1:length(rho)
    %     Temp0 = ADMMnm(M, rho(k), K, 0, (K/n), result1.X);
    %     [zz0,~] = principal_k(Temp0.X, K);
    %     [Acc_bound1(i,k), Nmi_bound1(i,k)] = rep_kmeans(zz0, class, lab, repk);
    % end
    % if i > 2
    %     rho = linspace(0.01,0.5,6)*10^5;
    % end
    for k = 1:length(rho)
        Temp = ADMMnm(M, rho(k), K, 0, (K/n), Intermedia{3});
        [zz,~] = principal_k(Temp.X, K);
        [Acc_bound2(i,k), Nmi_bound2(i,k)] = rep_kmeans(zz, class, lab, repk);    
    end
    %Acc(i,7) = max([Acc_bound1(i,:),Acc_bound2(i,:)]);
    Acc(i,7) = max(Acc_bound2(i,:));
    Nmi(i,7) = max(Nmi_bound2(i,:));
    %Nmi(i,7) = max([Nmi_bound1(i,:),Nmi_bound2(i,:)]);
    %%
    Acc_p = zeros(1,length(rho));
    Nmi_p = zeros(1,length(rho));
    for k = 1:length(rho)
        rez = ADMM_positive(M, rho(k), class, Lip, Intermedia{3});
        A = rez.X;
        [U,P] = principal_k(A, class);
        [Acc_p(k), Nmi_p(k)] = rep_kmeans(U, class, lab, repk);
    end
    Acc(i,8) = max(Acc_p(:));
    Nmi(i,8) = max(Nmi_p(:));

    fprintf([name{i},'\n']);
    format_convert([Acc(i,:),Nmi(i,:)]);

    %format_convert([Acc1,improve(1),Acc2,improve(2),Acc3,improve(3),Acc4,improve(4),Acc5,improve(5),Nmi1,improve(6),Nmi2,improve(7),Nmi3,improve(8),Nmi4,improve(9),Nmi5,improve(10)])
    
    %t = tiledlayout(1,7,'TileSpacing','Compact');
    
    % nexttile
    % imagesc(M)
    % 
    % nexttile
    % imagesc(result1.X)
    % 
    % nexttile
    % imagesc(result2.X)
    % 
    % nexttile
    % imagesc(P3)
    % 
    % nexttile
    % imagesc(Z)
    % 
    % nexttile
    % imagesc(A+A')
    % 
    % nexttile
    % imagesc(result6.X)

end

format_convert(Acc);
fprintf('\n')
format_convert(Nmi);
%format_convert([Acc,max(improve2(:,1:length(rho)),[],2),max(improve(:,1:length(rho)),[],2),Nmi,max(improve2(:,length(rho)+1:end),[],2),max(improve(:,length(rho)+1:end),[],2)]);




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
        %[acc(i),~,~] = AccMeasure(lab',idx');
        %nmii(i) = calMI(idx',lab');
        nmii(i) = nmi(lab', idx');
    end
    Acc = mean(acc);
    Nmi = mean(nmii);
end
