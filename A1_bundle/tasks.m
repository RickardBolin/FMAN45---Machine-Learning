load A1_data.mat

%% Calculations and plot for task 4
what = lasso_ccd(t, X, 0.1, zeros(1000,1));

title('Reconstruction plot with lambda=1.8', 'FontSize', 25)
xlabel('Time', 'FontSize', 18) 
ylabel('Noisy data', 'FontSize', 18)
hold on

% Original data
scatter(n, t, 70 ,'r', 'x');

% Regression
y = X*what;
scatter(n, y, 30, 'b');

% Interpolation
plot(ninterp, Xinterp*what, 'g')
legend({'Original data','Prediction', 'Interpolation'}, 'FontSize', 13)

%% Calculate how many non-negative weights are needed
non_zero = sum(what ~= 0);


%% Task 5 - Lasso-CV

lambda_max = max(abs(X'*t));
lambda_min = 0.5;
N_lambda = 50;
lambdavec = exp(linspace(log(lambda_min), log(lambda_max), N_lambda));
[wopt,lambdaopt,RMSEval,RMSEest] = lasso_cv(t, X, lambdavec, 10);
%% Plot
title('RMSE for different lambdas', 'FontSize', 25)
xlabel('Lambda', 'FontSize', 18) 
ylabel('Error', 'FontSize', 18)
hold on

plot(lambdavec, RMSEval, 'x');
plot(lambdavec, RMSEest, 'o');
plot([lambdaopt lambdaopt], [0, 4], '--');
xlim([lambda_min,lambda_max])
set(gca, 'XTick', unique([lambdaopt, get(gca, 'XTick')]));
set(gca,'FontSize',18)

legend({'RMSE for validation','RMSE for estimation', 'Optimal lambda'}, 'FontSize', 16)

%% Reconstruction plot with the optimal lambda
title('Reconstruction plot for lambda=2.0909', 'FontSize', 25)
xlabel('Time', 'FontSize', 18) 
ylabel('Noisy data', 'FontSize', 18)
hold on

% Original data
scatter(n, t, 70 ,'r', 'x');

% Regression
y = X*wopt;
scatter(n, y, 30, 'b');

% Interpolation
plot(ninterp, Xinterp*wopt, 'g')
legend({'Original data','Best prediction', 'Interpolation'}, 'FontSize', 16)

%% Task 6 - Calculate Multiframe Lasso Cross Validation for different lambdas
lambda_max = 0.03; % max(abs(Xaudio'*Ttrain));
lambda_min = 0.0001;
N_lambda = 50;
lambdavec = exp(linspace(log(lambda_min), log(lambda_max), N_lambda));
[wopt,lambdaopt,RMSEval,RMSEest] = multiframe_lasso_cv(Ttrain, Xaudio, lambdavec, 10);

%% Plot the RMSE for the different lambdas
title('RMSE for different lambdas', 'FontSize', 25)
xlabel('Lambda', 'FontSize', 22) 
ylabel('Error', 'FontSize', 22)
hold on

plot(lambdavec, RMSEval, 'x');
plot(lambdavec, RMSEest, 'o');
plot([lambdaopt lambdaopt], [0, 0.15], '--');
xlim([lambda_min,lambda_max])

legend({'RMSE for validation','RMSE for estimation', 'Optimal lambda = 0.0047'}, 'FontSize', 16)

%% Task 7 -  Denoise the test audio
Y_clean = lasso_denoise(Ttest, Xaudio, lambdaopt);

%% Save the denoised audio
save('denoised_audio','Y_clean','fs')