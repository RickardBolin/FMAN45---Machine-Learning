close all
clear all
load('models/network_trained_with_momentum.mat')

%% 6
% All filters for the first convolutional layer
for i=1:16
    figure(i)
    heatmap(net.layers{2}.params.weights(:,:,1,i))
end

%%
% Plot some misclassified images
x_test = loadMNISTImages('data/mnist/t10k-images.idx3-ubyte');
y_test = loadMNISTLabels('data/mnist/t10k-labels.idx1-ubyte');
y_test(y_test==0) = 10;
x_test = reshape(x_test, [28, 28, 1, 10000]);

pred = zeros(numel(y_test),1);
batch = 10;
for i=1:batch:size(y_test)
    idx = i:min(i+batch-1, numel(y_test));
    % note that y_test is only used for the loss and not the prediction
    y = evaluate(net, x_test(:,:,:,idx), y_test(idx));
    [~, p] = max(y{end-1}, [], 1);
    pred(idx) = p;
end

misclassified_idx = pred ~= y_test;
ground_truth = pred(misclassified_idx);
misclassified_label = y_test(misclassified_idx);
misclassified = x_test(:,:,:,misclassified_idx);
classes = [1:9 0];
siz = size(misclassified);
misclassified = reshape(misclassified, [28, 28, 1, siz(4)]);
for j=1:6
    figure
    imagesc(misclassified(:,:,1,j));
    colormap(gray);
    axis off;
    title(['Ground truth: ', num2str(classes((misclassified_label(j)))), '     Predicted: ', num2str(classes(ground_truth(j)))])
end

%%
% Plot confusion matrix
cm = confusionmat(y_test, pred);
heatmap(cm);

%%
% Calculate precision
sum_over_cols = sum(cm,2);
sum_over_rows = sum(cm);
true_positives = diag(cm);

precision = true_positives./sum_over_rows';
recall = true_positives./sum_over_cols;


%% 7
cifar10_starter()

%%
% All filters for the first convolutional layer
for i=1:16
    for j = 1:3
        figure
        heatmap(net.layers{2}.params.weights(:,:,j,i))
    end
end

%%
% Plot some misclassified images
[x_train, y_train, x_test, y_test, classes] = load_cifar10(2);
%%
y_test(y_test==0) = 10;
x_test = reshape(x_test, [32, 32, 3, 10000]);

pred = zeros(numel(y_test),1);
batch = 10;
for i=1:batch:size(y_test)
    idx = i:min(i+batch-1, numel(y_test));
    % note that y_test is only used for the loss and not the prediction
    y = evaluate(net, x_test(:,:,:,idx), y_test(idx));
    [~, p] = max(y{end-1}, [], 1);
    pred(idx) = p;
end

misclassified_idx = pred ~= y_test;
ground_truth = pred(misclassified_idx);
misclassified_label = y_test(misclassified_idx);
misclassified = x_test(:,:,:,misclassified_idx);
siz = size(misclassified);
%%
misclassified = reshape(misclassified, [32, 32, 3, siz(4)]);
for j=1:6
    figure
    imagesc(misclassified(:,:,:,j)/255);
    axis off;
    title(['Ground truth: ', num2str(classes{misclassified_label(j)}), '     Predicted: ', num2str(classes{ground_truth(j)})])
end

%%
% Plot confusion matrix
cm = confusionmat(y_test, uint8(pred));
heatmap(cm);

%%
% Calculate precision
sum_over_cols = sum(cm,2);
sum_over_rows = sum(cm);
true_positives = diag(cm);

precision = true_positives./sum_over_rows';
recall = true_positives./sum_over_cols;

