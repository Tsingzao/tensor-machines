%This function takes as input a training and a test set and trains tensor machines on the training
%and computes the generalization error.
%[error_test, error_train, solver_outputs, opt_outputs] = tm_solver(X, Y, Xt, Yt, type, options) 
%Inputs:
%  X, Y: training features and targets
%  Xt, Yt: test features and targets
%  type: 'regression' | 'bc'(binary classification)
%  options: options for tensor machines
%    the requred fields are s, solver, r, lambda, maxIter, alpha, verbosity (see "tensor_machines.m" for illustrations)
%Outputs:
%  error_test, error_train: test error and training error
%  solver_outputs: fields include trained weight vectors, running time
%  opt_outputs: output from the optimization sovler used

function [error_test, error_train, solver_outputs, opt_outputs] = tm_solver(X, Y, Xt, Yt, type, options) 

[n,d] = size(X);
nt = size(Xt,1);

%printing out parameters
fprintf('running tensor machines\n')
fprintf('data size: %d by %d\n', n, d)
fprintf('parameters: degree(%d)  rank(%d)  solver(%s)  lambda(%e)  maxIter(%d)  alpha(%f)\n',options.q,options.r,options.solver,options.lambda,options.maxIter,options.alpha)

%calling tensor machines on the training data
tic;
[solver_outputs.w, pred, opt_outputs] = tensor_machines(X, Y, type, options.r, options.solver, options.q, options.lambda, options.maxIter, options.alpha, options.verbosity);
solver_outputs.time_train = toc;

tic;
predt = get_tm_pred(solver_outputs.w, Xt, options.q, options.r, type); %get the predictions for the test data based on the learned model
solver_outputs.time_test = toc;

%computing the training and test error based on the type of the problem
switch type
    case 'bc'
        predt = sign(predt);
        pred = sign(pred); 
        error_test = mean(predt~=Yt);
        error_train = mean(pred~=Y);
    case 'regression'
        error_test = norm(Yt - predt)/norm(Yt);
        error_train = norm(Y - pred)/norm(Y);
    otherwise
        fprintf('Please enter a valid type!');
end
