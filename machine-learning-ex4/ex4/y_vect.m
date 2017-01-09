% vectorize y
% TODO: find a simpler way to vectorize
function [Y_t] = y_vect(y, num_labels)

for i = 1:length(y)
  for j = 1:num_labels
    if j == y(i)
      Y_t(i,j) = 1;
    else
      Y_t(i,j) = 0;
    endif
  endfor
 endfor
 
