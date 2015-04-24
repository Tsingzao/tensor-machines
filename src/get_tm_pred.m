%This function returns the prediction based on the learned tensor machines.
%Inputs:
%  X: original features
%  param: weights that are learned
%  q: degree
%  r: rank in the model
%  type: type of the problem in order to construct the correct predictions
%Output:
%  z: prediction for each row in X 

function z = get_tm_pred(param,X,q,r,type)

    [n,d] = size(X);
    r_vec = [1,r*ones(1,q-1)];

    b = param(1);
    w = param(2:end);
    w = reshape(w,d,length(w)/d);

    acc_sum = 0;
    w_count = 1;

    Z = b*ones(n,1);

    for i = 1:q
        for j = 1:r_vec(i)
            w_idx = w_count:(w_count+i-1);
            W = w(:,w_idx); %d-by-i
            XW = X*W; %n-by-i
            pxw = prod(XW,2); %n-by-1

            Z = Z + pxw;

            w_count = w_count+i;
        end
    end

    %the size of Z is n-by-1
    switch type
        case 'regression'
            z = Z;
        case 'bc'
            z = sign(Z);
    end

