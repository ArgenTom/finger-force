function my_poly = polyfit_tls(x_points, y_points, order)

    % Verify number of inputs
    if (nargin() ~= 3)
        error('Must have 3 inputs!');
    end
    
    % Verify size of inputs
    num_points = numel(x_points);
    if (numel(y_points) ~= num_points)
        error('Must have same number of points for x and y!');
    end
    
    % Condition data
    std_y = std(y_points);
    if (abs(std_y) > 1e-3)
        y_points = y_points / std_y;
    else
        % Ensure that later "reversal of conditioning" does nothing
        std_y = 1;
    end
    std_x = std(x_points);
    if (abs(std_x) > 1e-3)
        x_points = x_points / std_x;
    else
        % Ensure that later "reversal of conditioning" does nothing
        std_x = 1;
    end
    
    % Form matrix for independent variable
    A_matrix = ones(num_points, order+1);
    for orderIdx = 1:order
        A_matrix(:, orderIdx) = x_points.^(order - orderIdx + 1);
    end % orderIdx
    
    % Perform singular value decomposition
    [U, S, V] = svd([A_matrix y_points(:)], 'econ');
    
    % Extract the appropriate submatrices
    V12 = V(1:(order+1), (order+2):end);
    V22 = V((order+2):end, (order+2):end);
    
    if (det(V22) == 0)
        % If V22 is singular, there is no TLS solution
        error('No Total Least Squares solution exists!');
    else
        % Calculate the coefficients of the polynomial
        my_poly = -V12 / V22;
        
        % Transpose to get the same shape as Matlab's polyfit
        my_poly = my_poly';
        
        % Condition the coefficients to account for prior conditioning
        my_poly(1) = my_poly(1) * std_y / std_x;
        my_poly(2) = my_poly(2) * std_y;
    end

end % polyfit_tls
