%This script is for tensor machines with sfo solver.
%It accept any input weight vector and evaluates the corresponding objective (and gradient).
%It has minor changes compared to tm_f_g0.m.

function [f,g] = tm_f_g(param,X,q,r,type,Y,lambda)

    [n,d] = size(X);
    r_vec = [1,r*ones(1,q-1)];

    lambda = n*lambda;

    b = param(1);
    w = param(2:end);
    w = reshape(w,d,length(w)/d);
    nw = size(w,2);

    acc_sum = 0;
    w_count = 1;

    c = b;
    Z = c*ones(n,1);

    for i = 1:q
        for j = 1:r_vec(i)
            w_idx = w_count:(w_count+i-1);
            W = w(:,w_idx); %d-by-i
            XW = X*W; %n-by-i
            pxw = prod(XW,2); %n-by-1
            for l = 1:i
                idx = setdiff(1:i,l);
                bl(:,w_count+l-1) = prod(XW(:,idx),2);
            end

            Wsqaure = W.^2; % d-by-i
            norm_squares = sum(Wsqaure); %1-by-i

            acc_sum = acc_sum + sum(norm_squares);

            Z = Z + pxw;

            w_count = w_count+i;
        end
    end

    %the size of Z is n-by-1
    %the size of bl is n-by-nw
    switch type
        case 'regression'
            diff = Z-Y;
            f = norm(diff)^2/2;
        case 'bc'
            eyz = exp(-Y.*Z);
            diff = -Y.*eyz./(1+eyz);
            f = sum(log(1+eyz));
    end

    f = f + lambda*acc_sum/2;

    %the size of diff is n-by-1
    g_b = sum(diff)';
    g(1) = g_b;

    g_w = X'*(bsxfun(@times,diff,bl));
    g_w = g_w + lambda*w; 

    g(2:length(param)) = reshape(g_w,length(param)-1,1);
    