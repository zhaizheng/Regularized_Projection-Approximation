function re = ADMM_positive(A, lambda, k, lip, initA)

    n = size(A,1);

    rho = (sqrt(3))*lip*lambda;
    Z = zeros(size(A));

    [initU, initX] = principal_k(initA, k);
    X = initX;
    
    V = X + (Z/rho);
    tau = 2*lambda/rho;
    Y = opti(V, tau);

    Z =  Z + rho*(X-Y);
    iter = 1;
    error(iter) = norm(Y-X);
    [g(iter),h(iter)] = f(X,Y,Z,A,lambda,rho);
    
    while error(iter) >1.e-5 && iter<100
        [U, X] = principal_k((2.*A)+(rho.*Y)-Z, k);
        
        V = X + (Z/rho);
        Y = opti(V, tau);
        
        Z = Z + rho*(X-Y); 
        iter = iter+1;
        error(iter) = norm(X-Y);
        
        [g(iter),h(iter)] = f(X,Y,Z,A,lambda,rho);
        r(iter) = norm(X-initX);
    end
    re.X = X;
    re.Y = Y;
    re.Z = Z;
    re.g = g;
    re.h = h;
    re.r = r;
    re.U = U;
    re.initU = initU;

    function [g1,g2] = f(X,Y,Z,A,lambda,rho)
        g1 = norm(X-Y);
        %Tem = min(X-a,0);
        XY = X-Y;
        %g2 = norm(A-X,'fro')^2+(lambda*norm(Tem,'fro')^2);
        g2 = norm(A-X,'fro')^2+lambda*non_negative(Y)+norm(X-Y,'fro')^2*rho/2+Z(:)'*XY(:);
    end

    function re_t = non_negative(Y)
        As = min(Y,0);
        Bs = As.^2;
        re_t = sum(Bs(:));
    end

    function F = opti(V, tau)
        F = zeros(size(V));
        F(V>=0) = V(V>=0);
        F(V<0) = V(V<0)./(1+tau);
    end

end

    
% function [U,P] = principal_k(A, k)
%     [U, D]= eig(A);
%     [~, ind] = sort(diag(D),'descend');
%     U = U(:,ind(1:k));
%     P = U*U';
% end

function [U,P] = principal_k(A, k)
    [U, ~]= eigs(A,k);
    P = U*U';
end



