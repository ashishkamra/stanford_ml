function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%
% =========================================================================

% find all indices in 'y' with y == 1
pos = find(y==1);

% find all indices in 'y' with y == 0
neg = find(y==0);

% plot the y==1 data points in X
plot(X(pos,1), X(pos,2), 'x', 'LineWidth', 2, ...
      'MarkerSize', 5)

% plot the y==0 data points in X
plot(X(neg,1), X(neg,2), 'o', 'MarkerFaceColor', 'y', ...
      'MarkerSize', 5)

hold off;

end
