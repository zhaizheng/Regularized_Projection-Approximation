function re = ADMM_sparse(A, lambda, k, delta, lip)

    n = size(A,1);

    rho = (sqrt(3))*lip*lambda;
    Z = zeros(size(A));

    [initU, initX] = principal_k(A, k);
    X = initX;
    
    V = X + (Z/rho);
    tau = 2*lambda/rho;
    Y = opti(V, delta, tau);

    Z =  Z + rho*(X-Y);
    iter = 1;
    error(iter) = norm(Y-X);
    [g(iter),h(iter)] = f(X,Y,Z,A,delta,lambda,rho);
    
    while error(iter) >1.e-9 && iter<500
        [U, X] = principal_k((2.*A)+(rho.*Y)-Z, k);
        
        V = X + (Z/rho);
        Y = opti(V, delta, tau);
        
        Z = Z + rho*(X-Y); 
        iter = iter+1;
        error(iter) = norm(X-Y);
        
        [g(iter),h(iter)] = f(X,Y,Z,A,delta,lambda,rho);
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

    function [g1,g2] = f(X,Y,Z,A,delta,lambda,rho)
        g1 = norm(X-Y);
        %Tem = min(X-a,0);
        XY = X-Y;
        %g2 = norm(A-X,'fro')^2+(lambda*norm(Tem,'fro')^2);
        g2 = norm(A-X,'fro')^2+lambda*huber(Y, delta)+norm(X-Y,'fro')^2*rho/2+Z(:)'*XY(:);
    end

    function re = huber(Y, delta)
        U1 = Y.^2/(2*delta);
        U2 = abs(Y)-delta/2;
        UT = zeros(size(U1));
        for i = 1:size(U1,1)
            for j = 1:size(U1,2)
                if abs(Y(i,j))<delta
                    UT(i,j) = U1(i,j);
                else
                    UT(i,j) = U2(i,j);
                end
            end
        end
        re = sum(UT(:));
    end


    function F = opti(V, delta, tau)
        F = zeros(size(V));
        A1 = 2*delta/(2*delta+tau)*V;
        R = zeros([size(V),3]);
        C(:,:,1) = max(-delta,min(A1,delta));
        C(:,:,2) = max(V-tau/2,delta);%min(-delta,max(A4,delta));
        C(:,:,3) = min(V+tau/2,-delta);
        R(:,:,1) = (1+tau/(2*delta)).*(C(:,:,1).^2)-2*C(:,:,1).*V+V.^2;
        R(:,:,2) = (C(:,:,2)-V).^2+tau*(C(:,:,2))-tau*delta/2;
        R(:,:,3) = (C(:,:,3)-V).^2-tau*(C(:,:,3))-tau*delta/2;
        %[~,ind2] = min(R,[],3);
        for i = 1:size(A1,1)
            for j = 1:size(A1,2)
                [~,indz] = min([R(i,j,1),R(i,j,2),R(i,j,3)]);
                %F(i,j) = C(i,j,ind2(i,j));
                F(i,j) = C(i,j,indz);
            end
        end
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



