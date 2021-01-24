function [y, labels] = K_means_classification(data, target, K, C)
   labels = zeros(K,1);
   y = step_assign_cluster(data, C);
   for cluster = 1:K
       if sum(isnan(C(1,cluster))) == 0
           [amount, class] = groupcounts(target(y == cluster));
           [~, idxOfClass] = max(amount);
           labels(cluster) = class(idxOfClass);
       else
           labels(cluster) = NaN;
       end
   end   
end

function d = fxdist(x,C)
    K = size(C,2);
    d = zeros(K,1);
    for i = 1:K
        d(i) = norm(C(:,i) - x);
    end
end

function y = step_assign_cluster(X, C)
    N = size(X,2);
    y = zeros(N,1);
    for i = 1:N
        d = fxdist(X(:, i), C);
        [~, y(i)] = min(d);
    end
end