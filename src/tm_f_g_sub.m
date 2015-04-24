function [f,g] = tm_f_g_sub(param,data_idx,X,q,r,type,Y,lambda)

    X = X(data_idx,:);
    Y = Y(data_idx);
    [f,g] = tm_f_g(param,X,q,r,type,Y,lambda);

