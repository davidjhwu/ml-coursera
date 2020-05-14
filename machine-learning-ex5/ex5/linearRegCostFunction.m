function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
theta_reg = [zeros(1,1); theta(2:end,1)];

thetaSumSq = sum(theta_reg.^2);

h = X*theta; %predictions of hypothesis on all m examples (DW)
sqrErrors = (h-y).^2; %squared errors
J = 1/(2*m) * sum(sqrErrors) + ((lambda/(2*m))*thetaSumSq);



errors = h-y;
grad = X'*errors*((1/m)) + (theta_reg*(lambda/m));




% =========================================================================

grad = grad(:);

end
