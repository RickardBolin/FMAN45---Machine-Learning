load A1_data.mat

what = lasso_ccd(t, X, 4, zeros(1000,1));

%% Plot
title('Reconstruction plot', 'FontSize', 25)
xlabel('Time', 'FontSize', 18) 
ylabel('Noisy data', 'FontSize', 18)
hold on

% Original data
scatter(n, t, 20 ,'r');

% Regression
y = X*what;
scatter(n, y, 20, 'b');

% Interpolation
plot(ninterp, Xinterp*what)
legend({'Original data','Prediction', 'Interpolation'}, 'FontSize', 16)

%% Calculate how many non-negative weights are needed
non_zero = sum(what ~= 0);


%% Lasso-CV

lambda_max = max(abs(X'*t));
lambda_min = 0.5;
N_lambda = 10;
lambdavec = exp(linspace(log(lambda_min), log(lambda_max), N_lambda));
[wopt,lambdaopt,RMSEval,RMSEest] = lasso_cv(t, X, lambdavec, 3);

title('RMSE for different lambdas', 'FontSize', 25)
xlabel('Lambda', 'FontSize', 18) 
ylabel('Error', 'FontSize', 18)
hold on

plot(lambdavec, RMSEval, 'x');
plot(lambdavec, RMSEest, 'o');
plot([lambdaopt lambdaopt], [0, 10], '--');
legend({'RMSE for validation','RMSE for estimation', 'Optimal lambda'}, 'FontSize', 16)

%% 
title('Reconstruction plot', 'FontSize', 25)
xlabel('Time', 'FontSize', 18) 
ylabel('Noisy data', 'FontSize', 18)
hold on

% Original data
scatter(n, t, 20 ,'r');

% Regression
hold on
y = X*wopt;
scatter(n, y, 20, 'b');

% Interpolation
plot(ninterp, Xinterp*wopt)

legend({'Original data','Best prediction', 'Interpolation'}, 'FontSize', 16)



