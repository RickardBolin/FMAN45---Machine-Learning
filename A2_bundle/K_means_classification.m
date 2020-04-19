function [labels] = K_means_classification(target,y,K,C)
   labels = zeros(K,1);
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
