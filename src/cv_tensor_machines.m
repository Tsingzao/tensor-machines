%This script performs cross-validation for tensor machines.
%[err,err_std] = cv_tensor_machines(feats, target, type, alpha_range, lambdarange, rrange, solver, q, maxIter, verbosity)
%Inputs:
%  The following parameters for tensor machines should be fixed:
%    q, solver, maxIter, verbosity
%  The following parameters are needed cab be cross validated (so the inputs should be vectors):
%    alpha, lambda, r
%Outputs:
%  err: a 3D tensor consisting average error over partitions for all possible combination of parameters (alpha, lambda, rank)
%  err_std: standard deviations

function [err,err_std] = cv_tensor_machines(feats, target, type, alpha_range, lambdarange, rrange, solver, q, maxIter, verbosity)

if strcmp(verbosity, {'minimal'}) || strcmp(verbosity, {'off'})
    warning off
end

cvfolds = 5;
num_test_fold = 5;

ntrain = length(target);

lambdachoices = lambdarange;
numlambdas = length(lambdachoices);
err = zeros(length(alpha_range),numlambdas,length(rrange),cvfolds);

CVO = cvpartitionstub(ntrain, 'Kfold', cvfolds);

for pidx = 1:num_test_fold
    fprintf('on fold %d/%d\n', pidx, num_test_fold);
    trIdx = CVO.training(pidx);
    teIdx = CVO.test(pidx);
    training_feats = feats(trIdx, :);
    testing_feats = feats(teIdx, :);
    ytrain = target(trIdx);
    ytest = target(teIdx);
    ntrain_cv = size(ytrain, 1);
    ntest_cv = size(ytest, 1);

    for ridx = 1:length(rrange)
      for lambdaidx = 1:numlambdas
        for alphaidx = 1:length(alpha_range)
          switch verbosity
            case 'all'
              fprintf('On dataset %d, alpha: %f, lambda: %d and rank: %d\n', pidx, alphaidx, lambdaidx, ridx);
              v1 = 'all';
            case 'minimal'
              fprintf('.(%d,%d,%d,%d)',alphaidx, lambdaidx, ridx, pidx);
              v1 = 'off';
          end
    
          curlambda = lambdachoices(lambdaidx);
          curalpha = alpha_range(alphaidx);
          curr = rrange(ridx);

          w = tensor_machines(training_feats, ytrain, type, curr, solver, q, curlambda, maxIter, curalpha, v1); 
          predt = get_tm_pred(w, testing_feats, q, curr, type);

          switch type
            case 'regression'
                err(alphaidx, lambdaidx, ridx, pidx) = norm(ytest - predt)/norm(ytest);
            case 'bc'
                err(alphaidx, lambdaidx, ridx, pidx) = mean(ytest~=predt);
          end
        end
      end
    end

    if strcmp(verbosity, {'minimal'})
      fprintf('\n');
    end
end

err_std = std(err,0,4);
err = mean(err,4);  

if strcmp(verbosity, {'minimal'}) || strcmp(verbosity, {'off'})
    warning on
end
