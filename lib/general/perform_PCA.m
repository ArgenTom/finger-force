% PERFORM_PCA Calculates a principal component analysis of the data in
% data_array, which contains examples of data in each column, and each row
% represents different examples of a single data value.  The second input,
% break_point, represents a "knee point" at which to cut off further
% eigenvectors and eigenvalues.  If it is not included, all values are
% returned.  The third input, use_eckart, is a Boolean indicating that the
% Eckart-Young Theorem should be used to reduce the dimension of the
% problem.
% 
% Created by Thomas R. Grieve
% 3 April 2012
% University of Utah
% 

function [eigenvectors, eigenvalues, mean_data] = perform_PCA(data_array, break_point, use_eckart)

    if (nargin() == 1)
        break_point = 1;
        use_eckart = false;
    elseif (nargin() == 2)
        use_eckart = false;
    elseif (nargin() ~= 3)
        error('Must have 1, 2 or 3 inputs!');
    end
    
    % Determine the size of the inputs
    [num_values, num_examples] = size(data_array);
    
    % Find the centroid of the data & re-center
    mean_data = mean(data_array, 2);
    data_centered = zeros(num_values, num_examples);
    for exampleIdx = 1:num_examples
        data_centered(:,exampleIdx) = data_array(:,exampleIdx) - mean_data;
    end % exampleIdx
    clear data_array;
    
    % Calculate the combined matrix
    if (use_eckart)
        covariance_matrix = data_centered' * data_centered / num_examples;
    else
        covariance_matrix = data_centered * data_centered' / num_examples;
    end
    
    % Find the eigenvectors
    [evectors, evalues] = eig(covariance_matrix);
    
    % Extract the eigenvectors and eigenvalues to final outputs
    evalues = diag(evalues);
    if (use_eckart)
        % Condition the eigenvectors and eigenvalues to be normalized and
        % in greatest-to-least order
        eigenvalues = zeros(num_values, 1);
        eigenvectors = zeros(num_values, num_examples);
        for exampleIdx = 1:num_examples
            eigenvalues(exampleIdx) = evalues(num_examples - exampleIdx + 1);
            eigenvectors(:,exampleIdx) = data_centered * evectors(:, num_examples - exampleIdx + 1) / sqrt(num_examples * eigenvalues(exampleIdx));
        end % exampleIdx
    else
        % Only reverse the order
        eigenvalues = evalues(end:-1:1);
        eigenvectors = evectors(:,end:-1:1);
    end
    clear data_centered;
    
    if (break_point < 1)
        % Determine the number of eigenvalues to keep
        eigensum = cumsum(eigenvalues);
        num_eigs = find(eigensum > break_point*eigensum(end), 1);
        clear eigensum;
        
        % Adjust the eigenvectors/eigenvalues to keep only these
        eigenvectors = eigenvectors(:,1:num_eigs);
        eigenvalues = eigenvalues(1:num_eigs);
    end

end % perform_PCA
