close all
clear
load('models/network_trained_with_momentum.mat')

%% 6
% All filters for the first convolutional layer
figure
imagesc(net.layers{2}.params.weights(:,:,1,1))
title("The first filter", "FontSize", 25)

figure;
sgtitle("The other fifteen filters", "FontSize", 28)

for r = 1:3
    for c = 1:5
        subplot(3, 5, (r-1)*5 + c)
        imagesc(net.layers{2}.params.weights(:,:,1,(r-1)*5 + c + 1))
        axis off
    end
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
    title(['Ground truth: ', num2str(classes((misclassified_label(j)))), '     Predicted: ', num2str(classes(ground_truth(j)))], "FontSize", 25)
end

%%
% Plot confusion matrix
figure
cm = confusionmat(y_test, pred);
heatmap(cm);
xlabel('Actual')%, 'FontSize', 25)
ylabel('Predicted')%, 'FontSize', 25)

%%
% Calculate precision
sum_over_cols = sum(cm,2);
sum_over_rows = sum(cm);
true_positives = diag(cm);

precision = true_positives./sum_over_rows';
recall = true_positives./sum_over_cols;


%% 7
cifar10_starter()
%load('models/cifar10_new.mat')

%%
% All filters for the first convolutional layer
figure
for r = 1:4
    for c = 1:6
        subplot(4, 6, (r-1)*6 + c)
        imagesc(net.layers{2}.params.weights(:,:, min(idivide(int32((r-1)*6 + c), int32(8)) + 1, 3), mod((r-1)*6 + c, 8) + 1))
        axis off
    end
end
sgtitle("All 24 filters in the first layer", "FontSize", 28)

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
for j=1:3
    figure
    imagesc(misclassified(:,:,:,j)./max(max(misclassified(:,:,:,j))));
    axis off;
    title(['Ground truth: ', num2str(classes{misclassified_label(j)}), '     Predicted: ', num2str(classes{ground_truth(j)})], "FontSize", 25)
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

