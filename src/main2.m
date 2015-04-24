% Fit a TM to some artificially generated data
% NB: Don't forget to add minFunc to your matlab path if you use it as the solver
clear all; clc;

% load some artificially generated data 
identifier = 'sparse_degree6_target';
[type, X, Y, Xt, Yt] = genData(identifier);

% preprocessing impacts the performance of TMs
% the following line may help when using real datasets
%preproc_data

%set parameters
q = 6; % degree 
%solver = 'minFunc'; % use lbfgs solver (TM-Batch)
solver = 'sfo'; % use SFO solver (TM-SFO)
maxIter = 10; % number of iterations of the solver
verbosity = 'minimal';

% cross-validation ranges
alpha_range = [0.01,0.1]; % the norm of the random points used to initialize the TM parameters
lambdarange = [1e-8,1e-6,1e-4,1e-2]; % the regularization constant
rrange = [3,4,5,6];

% use cross validation on the training data to calculate errors and standard deviations
% the three dimensions of err and err_std correspond to alpha, lambda and rank, respectively.
[err,err_std] = cv_tensor_machines(X, Y, type, alpha_range, lambdarange, rrange, solver, q, maxIter, verbosity);

%call tensor machines using the best parameters from cross validation
options.q = q;
options.solver = solver;
options.maxIter = maxIter;
options.verbosity = verbosity;

[i,j,k] = ind2sub(size(err), find(err == min(err(:))));
options.alpha = alpha_range(i);
options.lambda = lambdarange(j);
options.r = rrange(k);
options.maxIter = maxIter*2; % run it for longer

% fit a TM: return its parameters and the test and train errors
[error_test, error_train, solver_outputs, opt_outputs] = tm_solver(X, Y, Xt, Yt, type, options);
fprintf('training error is %f\n', error_train)
fprintf('test error is %f\n', error_test)
fprintf('training time is %f (s)\n', solver_outputs.time_train)
fprintf('test time is %f (s)\n', solver_outputs.time_test)

% use the fitted TM to predict; plot predictions vs actual values
predt = get_tm_pred(solver_outputs.w, Xt, options.q, options.r, type);
[~, ascending_indices] = sort(Yt);
h = figure()
plot(Yt(ascending_indices), Yt(ascending_indices), 'g+', Yt(ascending_indices), predt(ascending_indices), 'r*');
legend('actual values', 'predicted values');
xlabel('Yt');
saveas(h,'recover.jpg')
