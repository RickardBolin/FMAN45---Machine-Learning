function [y,C] = K_means_clustering(X,K)

% Calculating cluster centroids and cluster assignments for:
% Input:    X   DxN matrix of input data
%           K   Number of clusters
%
% Output:   y   Nx1 vector of cluster assignments
%           C   DxK matrix of cluster centroids

[D,N] = size(X);

intermax = 50;
conv_tol = 1e-6;
% Initialize
C = repmat(mean(X,2),1,K) + repmat(std(X,[],2),1,K).*randn(D,K);
y = zeros(N,1);
Cold = C;

for kiter = 1:intermax

    % Step 1: Assign to clusters
    y = step_assign_cluster(X, C);
    
    % Step 2: Assign new clusters
    C = step_compute_mean(X, y, K);
        
    if fcdist(C,Cold) < conv_tol
        return
    end
    Cold = C;
end

end

function d = fxdist(x,C)
    K = size(C,2);
    d = zeros(K,1);
    for i = 1:K
        d(i) = norm(C(:,i) - x);
    end
end

function d = fcdist(C1,C2)
    d = norm(C1-C2);
end

function y = step_assign_cluster(X, C)
    N = size(X,2);
    y = zeros(N,1);
    for i = 1:N
        d = fxdist(X(:, i), C);
        [~, y(i)] = min(d);
    end
end

function C = step_compute_mean(X,y,K)
    C = zeros(size(X,1),K);
    for i = 1:K
        C(:,i) = mean(X(:,y == i), 2);
    end
end

