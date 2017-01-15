function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%vectorize y lables
Y_t = y_vect(y,num_labels);

%%% Feedforward

% first layer
X = [ones(m, 1) X]; % Add the bias
A_1 = X;

% hidden layer
Z_2 = A_1*Theta1';
A_2 = sigmoid(Z_2);
A_2 = [ones(m,1) A_2]; % Add the bias

% output layer
Z_3 = A_2*Theta2';
A_3 = sigmoid(Z_3);

%% Implementing backpropagation algorithm

% for every training example, do the following
for t = 1:m

% STEP:1 - feedforward (from above)

  % a_1 is the input layer
  % X already has the bias added earlier in the code
  a_1 = X(t,:); 
  
  % hidden layer activation
  z_2 = Z_2(t,:);
  a_2 = A_2(t,:);
  
  %o/p layer activation
  z_3 = Z_3(t,:);
  a_3 = A_3(t,:);
  
% STEP:2 - calculate the error terms

  % the output layer error term
  y = Y_t(t,:);
  delta_3 = a_3 - y;
  
  %hidden layer error term
  delta_2 = delta_3*Theta2;
  delta_2 = delta_2(2:end); % ignore the bias term in delta_2
  delta_2 = delta_2.*sigmoidGradient(z_2);
  
  %calculate the gradient
  Theta2_grad = Theta2_grad + delta_3'*a_2;
  Theta1_grad = Theta1_grad + delta_2'*a_1;

  %calculate the cost  
  J = J + (y*log(a_3') + (1 - y)*log(1 - a_3'));
endfor

%cost regularization. slice the first column out of each Theta
Theta1_s = Theta1(:,2:end);
Theta2_s = Theta2(:,2:end);

% regularization term is sum of squares of all values of Theta1_s and Theta2_s
regu_t = (lambda/(2*m))*(sum(sum(Theta1_s.^2)) + sum(sum(Theta2_s.^2)));
J = (-1/m)*J + regu_t;

%gradient regularization
% add a column vector of zeros (do not regularize the bias term
Theta1_r = [zeros(size(Theta1_s,1),1) Theta1_s];
Theta2_r = [zeros(size(Theta2_s,1),1) Theta2_s];

Theta2_grad = (1/m)*Theta2_grad + (lambda/m)*Theta2_r;
Theta1_grad = (1/m)*Theta1_grad + (lambda/m)*Theta1_r;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
