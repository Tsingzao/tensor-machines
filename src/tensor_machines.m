% This function computes a polynomial approximation to the target by using
% tensor machines w/ separate features for each degree
% [w, z, opt_outputs] = tensor_machines(X, Y, type, r, solver, q, lambda, maxIter, verbosity, alpha)
% Input:
%   X, Y: feature matrix and target
%   type: 'regression' | 'bc'(binary classification)
%   r: rank parameter in tensor machines
%   solver: 'minFunc' | 'sfo'
%   q: degree of the polynomial used
%   lambda: regularization parameter 
%   maxIter: number of iterations in minFunc or number of epochs in SFO
%   alpha: scaling factor of the initial weights
%   verbosity: 'iter' | 'final' | 'off'
% Output:
%   w: weights returned by the optimization solver which can be used to compute predictions for new points
%   z: predictions of X based on w
%   opt_outputs: outputs from the optimization solver which contains convergence information
% Solvers:
%   In this function, two publicly available solvers can be used for solving the optimization problem, namely, minFunc and SFO.
%   They can be downoaded via the following links:
%     minFunc: http://www.cs.ubc.ca/~schmidtm/Software/minFunc.html
%     SFO: https://github.com/Sohl-Dickstein/Sum-of-Functions-Optimizer
%   In the Tensor Machines package, a version of SFO is included.

function [w, z, opt_outputs] = tensor_machines(X, Y, type, r, solver, q, lambda, maxIter, alpha, verbosity)

  [n,d] = size(X); 

  %computing how many variables at total
  nv = 1+d+(q-1)*(q+2)*r*d/2;

  switch solver
    case 'minFunc'

      switch verbosity
      case 'all'
        verb = 'iter';
      case 'minimal'
        verb = 'final';
      case 'off'
        verb = 'off';
      end

      options = struct('Method', 'lbfgs', 'Corr', 2000, 'maxIter', maxIter, 'display', verb, 'progTol', 1e-8, 'MaxFunEvals', 8000); 

      w0 = alpha*randn([nv,1]); %setting initial weights
      [w, ~, ~, opt_outputs] = minFunc(@tm_f_g0, w0, options, X, q, r, type, Y, lambda);

    case 'sfo'

      w0 = alpha*randn([nv,1]); %initializing weights
      N = max(30,floor(sqrt(n)/10)); %setting size of mini-batches
      randp = randperm(n);
      sub_refs = cell(N,1);
      for i = 1:N
         sub_refs{i} = randp(i:N:n);
      end 

      optimizer = sfo(@tm_f_g_sub,w0,sub_refs,X,q,r,type,Y,lambda);

      switch verbosity
      case 'off'
        optimizer.display = 0;
      case 'minimal'
        optimizer.display = 1;
      case 'all'
        optimizer.display = 2;
      end

      w = optimizer.optimize(maxIter);
      opt_outputs = optimizer;

    otherwise
      fprintf('Please enter a valid solver!')
  end 

  z = get_tm_pred(w,X,q,r,type);

