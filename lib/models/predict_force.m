function [forces, use_image] = predict_force(current_image, prediction_model, options)

    % Determine prediction model type
    if (isfield(prediction_model,'slope_matrix'))
        [forces, use_image] = predict_linearized_sigmoid(current_image, prediction_model, options);
    elseif (isfield(prediction_model,'linear_weights'))
        [forces, use_image] = predict_eigennail(current_image, prediction_model);
    elseif (isfield(prediction_model,'param_weights'))
        [forces, use_image] = predict_AAMFinger(current_image, prediction_model);
    elseif (isfield(prediction_model,'cutoff_index'))
        [forces, use_image] = predict_pls(current_image, prediction_model);
    else
        error('Cannot determine Prediction Model type!');
    end

end % predict_force

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Perform Linearized Sigmoid force prediction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [forces, use_image] = predict_linearized_sigmoid(current_image, prediction_model, options)

    % Process the options
    if (isfield(options,'frac_useful_pixels'))
        frac_useful_pixels = options.frac_useful_pixels;
    else
        frac_useful_pixels = false;
    end
    
    % Extract data from prediction model structure
    slope_matrix = prediction_model.slope_matrix;
    pixel_offset = prediction_model.pixel_offset;
    Sigma = prediction_model.covariance_matrix;
    saturation = prediction_model.saturation;
    use_pixel = prediction_model.use_pixel;
    
    % Verify that size of input image is consistent with model
    if (numel(current_image) ~= length(use_pixel))
        error('Input image is not consistent with model!');
    end
    
    % Remove unused pixels from input image and reshape to column vector
    current_image = reshape(current_image(use_pixel), sum(use_pixel), 1);
    
    % Determine size of inputs
    num_pixels = length(current_image);
    num_forces = size(slope_matrix, 2);
    
    % Determine which pixels are within saturation limits
    useful_pixels = ((current_image >= saturation(:,1)) & (current_image <= saturation(:,2)));
    
    % Check for images with too few pixels in the useful range
    if ((sum(useful_pixels) < frac_useful_pixels*num_pixels) || (sum(useful_pixels) < num_forces))
        % Skip images with too few pixels in the useful range
        forces = zeros(1,num_forces);
        use_image = false;
    else
        % Apply saturation limits to pixels
        recentered_image = current_image;
        low_sat = find(current_image < saturation(:,1));
        hi_sat = find(current_image > saturation(:,2));
        recentered_image(low_sat) = saturation(low_sat,1);
        recentered_image(hi_sat) = saturation(hi_sat,2);
        
        % Re-center the image by removing the pixel offset
        recentered_image = recentered_image - pixel_offset;
        
        % Then use f = inv(B'*invS*B)*B'*invS*(p-p0) to predict force
        right_matrix = slope_matrix' * (Sigma \ recentered_image);
        left_matrix = slope_matrix' * (Sigma \ slope_matrix);
        if (rank(left_matrix) < num_forces)
            keyboard;
        end
        forces = (left_matrix \ right_matrix)';
        use_image = true;
    end

end % predict_linearized_sigmoid

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Perform EigenNail force prediction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [forces, use_image] = predict_eigennail(current_image, prediction_model)

    % Verify that size of input image is consistent with model
    num_pixels = length(prediction_model.centroid);
    if (numel(current_image) ~= num_pixels)
        error('Input image is not consistent with model!');
    end
    current_image = reshape(current_image, num_pixels, 1);
    centroid_image = reshape(prediction_model.centroid, num_pixels, 1);
    
    % Subtract the centroid
    recentered_image = current_image - centroid_image;
    
    % Find the weights of this image in the EigenNail space
    weight_matrix = recentered_image'*prediction_model.eigennails;
    
    % Apply the linear weights to estimate the force
    forces = [1 weight_matrix]*prediction_model.linear_weights;
    
    % Define the second output as true (since the EigenNail model is used)
    use_image = true;

end % predict_eigennail

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Perform AAMFinger (Gray-Level Parameters) force prediction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [forces, use_image] = predict_AAMFinger(current_image, prediction_model)

    % Verify that size of input image is consistent with model
    num_params = size(prediction_model.param_weights,1) - 1;
    if (numel(current_image) ~= num_params)
        error('Input image is not consistent with model!');
    end
    current_image = reshape(current_image, 1, num_params);
    
    % Apply the linear weights to estimate the force
    forces = [1 current_image]*prediction_model.param_weights;
    
    % Define the second output as true (since the AAMFinger model is used)
    use_image = true;

end % predict_AAMFinger

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Perform Partial Least Squares force prediction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [forces, use_image] = predict_pls(current_image, prediction_model)

    % Verify that size of input image is consistent with model
    cutoff_index = prediction_model.cutoff_index;
    num_pixels = length(prediction_model.mean_X);
    if (numel(current_image) ~= num_pixels)
        error('Input image is not consistent with model!');
    end
    
    % Remove mean and standard deviation from pixel matrix
    p_scaled = (current_image - prediction_model.mean_X) ./ prediction_model.std_X;
    f_estimate = [1 p_scaled(:, 1:cutoff_index)] * prediction_model.beta_values;
    forces = f_estimate .* prediction_model.std_Y + prediction_model.mean_Y;
    
    % Define the second output as true (since the AAMFinger model is used)
    use_image = true;

end % predict_pls
