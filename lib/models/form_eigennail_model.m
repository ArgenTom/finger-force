%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Form the EigenNail Magnitude Model, given then matrix of forces and the
% matrix of pixels, as well as the options (number of eigenvalues or cutoff
% percentage to use, whether or not to ignore the first eigenvector, and
% whether to produce verbose output).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function prediction_model = form_eigennail_model(forces, pixels, options)

    % Verify inputs
    if (nargin() == 2)
        [num_eigs, eigen_cutoff, ignore_first, verbose, debug_mode] = unpack_options([]);
    elseif (nargin() == 3)
        [num_eigs, eigen_cutoff, ignore_first, verbose, debug_mode] = unpack_options(options);
    else
        error('Must have 2 or 3 inputs!');
    end
    
    % Determine size of inputs
    [num_images, num_pixels] = size(pixels);
    if (debug_mode)
        fprintf('Forming EigenNail Magnitude Model');
    end
    
    % Determine whether to use the Eckart-Young Theorem to reduce the
    % dimension of the problem
    use_eckart = (num_pixels > num_images);
    
    % Perform PCA on the pixel matrix
    [eigennails, eigenvalues, centroid] = perform_PCA(pixels', eigen_cutoff, use_eckart);
    if ((verbose) || (debug_mode))
        fprintf('.');
    end
    
    % Adjust values to compensate for differing size/shape expectations
    if (isinf(num_eigs))
        num_eigs = length(eigenvalues);
    else
        eigennails = eigennails(:,1:num_eigs);
        eigenvalues = eigenvalues(1:num_eigs);
    end
    
    % Calculate the eigenvector weights of each image
    weight_matrix = zeros(num_images, num_eigs);
    for imageIdx = 1:num_images
        % Find the projection of the training image onto the eigenfinger basis
        weight_matrix(imageIdx,:) = (pixels(imageIdx,:) - centroid')*eigennails;
    end
    if ((verbose) || (debug_mode))
        fprintf('.');
    end
    
    % Use Multiple Linear Regression (i.e., Least Squares) to find the
    % linear weights that correlate the eigenvector weights to the forces
    if (ignore_first)
        % Ignore the first EigenNail
        linear_weights = [ones(num_images,1) weight_matrix(:,2:end)] \ forces;
        linear_weights = [linear_weights(1,:); 0 0 0; linear_weights(2:end,:)];
    else
        % Ignore the first EigenNail
        linear_weights = [ones(num_images,1) weight_matrix] \ forces;
    end
    if ((verbose) || (debug_mode))
        fprintf('.');
    end
    
    % Assign outputs
    prediction_model.eigennails = eigennails;
    prediction_model.eigenvalues = eigenvalues;
    prediction_model.centroid = centroid;
    prediction_model.linear_weights = linear_weights;
    if (debug_mode)
        fprintf('Done!\n');
    end

end % form_eigennail_model

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Process the input parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [num_eigs, eigen_cutoff, ignore_first, verbose, debug_mode] = unpack_options(options)

    % Assign default options, if they do not exist
    if (isfield(options,'num_eigs'))
        num_eigs = options.num_eigs;
        eigen_cutoff = 1;
    else
        num_eigs = Inf;
        if (isfield(options,'eigen_cutoff'))
            eigen_cutoff = options.eigen_cutoff;
        else
            eigen_cutoff = 0.99;
        end
    end
    if (isfield(options,'ignore_first'))
        ignore_first = options.ignore_first;
    else
        ignore_first = false;
    end
    if (isfield(options,'verbose'))
        verbose = options.verbose;
    else
        verbose = false;
    end
    if (isfield(options,'debug_mode'))
        debug_mode = options.debug_mode;
    else
        debug_mode = false;
    end

end % process_inputs
