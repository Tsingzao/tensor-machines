% CVO = mycvpartition(N, type, K)
%
% A partial reimplementation of MATLAB's statistics toolbox's cvpartition.
% For now, it implements the equivalents of the calls
%  cvpartition(N, 'Kfold')
%  cvpartition(N, 'Kfold', K)

function CVO = mycvpartition(N, type, K)

switch upper(type)
    case 'KFOLD'
        if nargin == 2
            K = 10;
        end
        if N < K
            error('N should be at least K');
        end
        CVO.NumTestSets = K;
        
        chunksize = ones(K,1);
        if mod(N,K) == 0
            chunksize = N/K*ones(K,1);
        else
            chunksize(2:end) = floor(N/K);
            chunksize(1) = N - floor(N/K)*(K-1);
        end
        subsetindices = mat2cell(randperm(N), 1, chunksize);
        
        CVO.test = @(idx) ismember(1:N, subsetindices{idx});
        CVO.training = @(idx) ~CVO.test(idx);
    otherwise
        error('Unimplemented');
end

end

