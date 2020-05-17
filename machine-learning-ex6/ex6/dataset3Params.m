function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
error_init = 42069

for C_val = values
    for sigma_val = values

        
        model = svmTrain(X, y, C_val, @(x1, x2) gaussianKernel(x1, x2, sigma_val));
        
        predictions = svmPredict(model, Xval);
        error = mean(double(predictions ~= yval)); 
                % error defined as fraction of crossvalidation examples classified incorrectly
        if (error < error_init)
            C_final = C_val;
            sigma_final = sigma_val;
            error_init = error;
            fprintf('found new C, sigma = %f, %f with error = %f', C_final, sigma_final, error_init)
        end
    endfor
endfor
        C = C_final;
        sigma = sigma_final;
 
%=====================================================================
end
