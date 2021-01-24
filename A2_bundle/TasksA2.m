clear
close all
load A2_data.mat

%% Task T2
x = [-2 -1 1 2];
y = [1 -1 -1 1];
K = [20 6 2 12; 6 2 0 2; 2 0 2 6; 12 2 6 20];
s = 0;
for i = 1:4
    for j = 1:4
        s = s + (y(i)*y(j)*K(i,j));
    end
end
alpha = 4/s;


%% TASK E1

K = 2;
d = 2;
norm_mean = mean(train_data_01,2);

norm_data = (train_data_01 - norm_mean);
norm_test_data = (test_data_01 - norm_mean);
[U, S, V] = svd(norm_data);
%%
proj = U(:,1:d)'*norm_data;
proj_test = U(:,1:d)'*norm_test_data;
%%

figure
title('Projection of the training data onto two dimensions', 'FontSize', 20)
hold on
c_map = parula(12);
proj = proj';

plot(proj(train_labels_01==0,1), proj(train_labels_01==0,2), 'o', 'Color', 'r')
plot(proj(train_labels_01==1,1), proj(train_labels_01==1,2), '*', 'Color', 'b')

legend('Class 0', 'Class 1')
proj = proj';
%% TASK E2
K = 5;
[y,C] = K_means_clustering(proj,K);

%%
figure
title(['Visualization of ', num2str(K), ' centroids with corresponding clusters' ], 'FontSize', 20)
hold on
c_map = parula(12);
proj = proj';
Markers = {'s','o','*','x','d'};
for i = 1:K
    plot(proj(y==i,1), proj(y==i,2), Markers{mod(i,4)+1}, 'Color', c_map(4*i,:))
end
proj = proj';

scatter(C(1,:),C(2,:), 200, 'black', 'filled')
axis([-10, 6, -6, 8])
legend('Cluster 1', 'Cluster 2')

%% TASK E3
figure
title('Visualization of centroid as images')

for i = 1:K
    im_zero = C(1,i)*U(:, 1) + C(2,i)*U(:, 2);
    subplot(ceil(K/3),min(3, K),i)
    imshow(reshape(im_zero, 28, 28));
    title(['Centroid ', num2str(i) , ' as image'], 'FontSize', 14)
end


%% TASK E4
[~,labels] = K_means_classification(proj, train_labels_01, K, C);
%%
Legend = cell(K,1);
counter = 1;
for iter=1:K
    if ~isnan(labels(iter))
       Legend{counter}=strcat('Label: ', num2str(labels(iter)));
       counter = counter + 1;
    end
end
legend(Legend(1:counter-1))

%% Correct and fail for training data
correct = zeros(K,1);
fail = zeros(K,1);
for i = 1:K
    correct(i) = sum(train_labels_01(y == i) == labels(i));
    fail(i) = sum(train_labels_01(y == i) ~= labels(i));
end

%% Table for Training data
M = zeros(K,5);
M(:,1) = 1:K;
for i = 1:K
    for j = 1:2
        M(i,j+1) = sum(train_labels_01(y == i) == j-1);
    end
end
M(:,4) = labels;
M(:,5) = fail;
Ntrain = size(proj,2);
sumMisclassified = sum(fail);
failRate = 100*sumMisclassified/Ntrain;

%%
[y_test, labels] = K_means_classification(proj_test, test_labels_01, K, C);
%% Correct and fail for test data 
correct = zeros(K,1);
fail = zeros(K,1);
for i = 1:K
    correct(i) = sum(test_labels_01(y_test == i) == labels(i));
    fail(i) = sum(test_labels_01(y_test == i) ~= labels(i));
end

%% Table for Test data
M = zeros(K,5);
M(:,1) = 1:K;
for i = 1:K
    for j = 1:2
        M(i,j+1) = sum(test_labels_01(y_test == i) == j-1);
    end
end
M(:,4) = labels;
M(:,5) = fail;
Ntrain = size(proj,2);
sumMisclassified = sum(fail);
failRate = 100*sumMisclassified/Ntrain;


%% TASK E6 - SVM
%Training data:
model = fitcsvm(train_data_01', train_labels_01);
pred_value_train = predict(model, train_data_01');
correct_train = sum(pred_value_train == train_labels_01);

total_ones_train = sum(train_labels_01 == 1);
correct_ones_train = sum(train_labels_01(pred_value_train == 1) == 1);
total_zeros_train = sum(train_labels_01 == 0);
correct_zeros_train = sum(train_labels_01(pred_value_train == 0) == 0);

% Test data:
pred_value = predict(model, test_data_01');
correct_test = sum(pred_value == test_labels_01);

total_ones_test = sum(test_labels_01 == 1);
correct_ones_test = sum(test_labels_01(pred_value == 1) == 1);
total_zeros_test = sum(test_labels_01 == 0);
correct_zeros_test = sum(test_labels_01(pred_value == 0) == 0);


%% TASK E7 
sigma = 0.2;
beta = sqrt(1/sigma^2);
model = fitcsvm(train_data_01', train_labels_01, 'KernelFunction', 'gaussian', 'KernelScale', beta);
%% Training data
pred_value_train = predict(model, train_data_01');
%%
correct_train = sum(pred_value_train == train_labels_01);
total_ones_train = sum(train_labels_01 == 1);
total_zeros_train = sum(train_labels_01 == 0);
correct_ones_train = sum(train_labels_01(pred_value_train == 1) == 1);
correct_zeros_train = sum(train_labels_01(pred_value_train == 0) == 0);

%% Test data
pred_value = predict(model, test_data_01');
correct_test = sum(pred_value == test_labels_01);
fail_test = sum(pred_value ~= test_labels_01);

total_ones_test = sum(test_labels_01 == 1);
correct_ones_test = sum(test_labels_01(pred_value == 1) == 1);
total_zeros_test = sum(test_labels_01 == 0);
correct_zeros_test = sum(test_labels_01(pred_value == 0) == 0);

misclassificationRate = 1 - ((correct_ones_test + correct_zeros_test)/2115);
