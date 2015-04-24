norms = sqrt(sum(X.^2,1));
X(:,find(norms)) = bsxfun(@rdivide,X(:,find(norms)),norms(find(norms)));
Xt(:,find(norms)) = bsxfun(@rdivide,Xt(:,find(norms)),norms(find(norms)));

norms = sqrt(sum(X.^2,2));
X = bsxfun(@ldivide,norms,X);
norms = sqrt(sum(Xt.^2,2));
Xt = bsxfun(@ldivide,norms,Xt);

