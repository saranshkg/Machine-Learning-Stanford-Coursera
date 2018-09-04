function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    theta -= alpha/m * X' * (X*theta - y);
    
    %{
    len_theta = length(theta);
    
    theta_tmp = theta;    

    for j = 1:len_theta
        sum_val = 0;
	
	for k = 1:m
            sum_val += (X(k,:) * theta - y(k,:)) * X(k,j);
        end

        theta_tmp(j,:) -= alpha/m * sum_val
    end

    theta = theta_tmp;
    %}

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
