clear
tic
s = [25 25 25];
noise = [0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8];
signal = [0.51,0.57,0.63,0.69];
rep = 10;
lambda = 0.04:0.04:1;
repk = 100;
for i = 1:length(signal)
    for k = 1:length(lambda)
        Acc = zeros(1,rep);
        Nmi = zeros(1,rep);
        for j = 1:rep
            [S, ~, label] = data_generation_unbal(signal(i), noise(i), s);
            n_iter = 100000;
            K = 3; class = K; lab = label;
            [A, ~, ~] = CLR_zz(S, lambda(k), K, n_iter);
            [U5,~] = principal_k(A+A', K);
            [Acc(j), Nmi(j)] = rep_kmeans(U5, class, lab, repk);
        end
        ac(k) = mean(Acc);
        mi(k) = mean(Nmi);
    end
    %%
    plot(lambda,ac)
    hold on
    plot(lambda,0.968*ones(size(lambda)));
    hold on
    plot(lambda,0.876*ones(size(lambda)));
    hold on
    plot(lambda,0.665*ones(size(lambda)));
    hold on
    plot(lambda,0.516*ones(size(lambda)));
    hold on
end
fprintf('Time cost:%f\n',toc);


function [Acc, Nmi] = rep_kmeans(F, class, lab, repk)
    F = diag(1./(sqrt(sum(F.^2,2))+eps))*F;
    for i = 1:repk
        idx = kmeans(real(F), class);
        [acc(i),~,~] = AccMeasure(lab',idx');
        nmii(i) = nmi(lab', idx');
    end
    Acc = mean(acc)/100;
    Nmi = mean(nmii);
end




    
function [U,P] = principal_k(A, k)
    [U, D]= eig(A);
    [~, ind] = sort(diag(D),'descend');
    U = U(:,ind(1:k));
    P = U*U';
end




