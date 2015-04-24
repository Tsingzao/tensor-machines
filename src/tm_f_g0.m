%This function is for tensor machines with minFunc solver.
%It evaluates the function value (and gradient) for any input iterate.

function varargout = tm_f_g0(param,X,q,r,type,Y,lambda)

    [n,d] = size(X);
    r_vec = [1,r*ones(1,q-1)];

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
            bl(:,w_idx) = repmat(pxw,1,i)./XW;

            Wsqaure = W.^2;
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
            f = norm(diff)^2/n/2;
        case 'bc'
            eyz = exp(-Y.*Z);
            diff = -Y.*eyz./(1+eyz);
            f = mean(log(1+eyz));
    end

    %adding regularizations
    f = f + lambda*acc_sum/2;

    varargout{1} = f;

    %the size of diff is n-by-k
    if nargout > 1
      g_b = mean(diff)';
      g(1) = g_b;

      g_w = X'*(bsxfun(@times,diff,bl))/n;
      g_w = g_w + lambda*w; 

      g(2:length(param)) = reshape(g_w,length(param)-1,1);

      varargout{2} = g';
    end

