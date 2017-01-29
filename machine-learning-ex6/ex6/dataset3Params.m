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

%initialize the params
p_error_min = 1;
c = 0.01;
while c < 100
  sig = 0.01;
  while sig < 100
    % train SVM with C = c, and sigma = sig
    model= svmTrain(X, y, c, @(x1, x2) gaussianKernel(x1, x2, sig));
    
    %calculate the predictions
    predictions = svmPredict(model, Xval);
    
    %calculate the prediction error
    p_error = mean(double(predictions ~= yval));
    
    % choose the params with the min prediction error
    if p_error < p_error_min
      p_error_min = p_error;
      C = c;
      sigma = sig;
      %printf('p_error_min = %f', p_error_min);
    endif
    
    % try next value of sig
    sig = 3*sig;
  endwhile
  %try next value of c
  c = 3*c;
endwhile

%printf('C = %f',C);
%printf('sigma = %f',sigma);






% =========================================================================

end
