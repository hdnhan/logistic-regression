close all
clear
clc

data = xlsread('data/test.csv');
X = data(:, 2:end);
y = data(:, 1);
clear data
X = featureScaling(X);

X = [ones(size(y)), X];
ntrain = round(0.8 * length(y));

Xtrain = X(1:ntrain, :);
ytrain = y(1:ntrain);
Xtest = X(ntrain + 1:end, :);
ytest = y(ntrain + 1:end);
clear X y

tic
eta = 0.05;
lambda = 0;
theta = regularized_logistic_regression(Xtrain, ytrain, eta, lambda);
accuracy = mean((sigmoid(Xtest * theta) >= 0.5) == ytest)
toc

tic
eta = 0.05;
lambda = 1000;
theta = regularized_logistic_regression(Xtrain, ytrain, eta, lambda);
accuracy = mean((sigmoid(Xtest * theta) >= 0.5) == ytest)
toc

tic
eta = 0.1;
lambda = 0;
theta = regularized_logistic_regression(Xtrain, ytrain, eta, lambda);
accuracy = mean((sigmoid(Xtest * theta) >= 0.5) == ytest)
toc


tic
eta = 0.1;
lambda = 1000;
theta = regularized_logistic_regression(Xtrain, ytrain, eta, lambda);
accuracy = mean((sigmoid(Xtest * theta) >= 0.5) == ytest)
toc