function coeffs = naive_total_least_squares(x_values, y_values)

    % Determine data size
    num_points = length(x_values);
    num_vars = 2;
    
    % Create matrix
    A = [ones(num_points, 1) x_values y_values];
    
    % Perform SVD
    [U,S,V] = svd(A,0);
    
    % Extract the appropriate blocks of V
    V_ab = V(1:num_vars, (num_vars+1):end);
    V_bb = V((num_vars+1):end, (num_vars+1):end);
    
    % Calculate the coefficients
    coeffs = -V_ab / V_bb;

end % naive_total_least_squares
