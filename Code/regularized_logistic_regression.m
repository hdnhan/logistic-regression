function theta = regularized_logistic_regression(X, y, eta, lambda, theta0, tol, maxit)
    if nargin < 7
        maxit = max(size(y, 1), 10000);
    end
    if nargin < 6
        tol = 1e-6;
    end
    if nargin < 5
        theta0 = zeros(size(X, 2), 1);
    end
    if nargin < 4
        lambda = 0;
    end
    if nargin < 3
        error('Not enough input arguments, should have at least (X, y, eta)');
    end

    % number of training examples
    m = length(y);
    theta = theta0;
    v0 = zeros(size(theta));

    check_theta_after = 100;
    gamma = 0.9;
    cnt = 1;
    while cnt < maxit
        reg = [0; theta0(2 : end)];
        h_theta = sigmoid(X * theta0);
        grad = 1 / m * X' * (h_theta - y) + lambda / m * reg;
        v = gamma * v0 + eta * grad;
        theta = theta0 - v;
        if mod(cnt, check_theta_after) == 0
            %sqrt(sum((theta - theta0) .^ 2))
            if sqrt(sum((theta - theta0) .^ 2)) < tol
                break
            end
        end
        v0 = v;
        theta0 = theta;
        cnt = cnt + 1;
    end
    cnt

end
