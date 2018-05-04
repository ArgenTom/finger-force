% predict_forces_on_dataset Predicts forces on the dataset contained in
%   pixels using the model in prediction_model.
% 

function [forces_training, use_image] = predict_forces_on_dataset(pixels, prediction_model, options)

    % Verify inputs
    if (nargin() == 2)
        options = [];
    elseif (nargin() ~= 3)
        error('Must have 2 or 3 inputs!');
    end
    num_images = size(pixels, 1);
    
    % Extract data from prediction model structure
    if (isfield(prediction_model,'slope_matrix'))
        num_forces = size(prediction_model.slope_matrix,2);
    elseif (isfield(prediction_model,'linear_weights'))
        num_forces = size(prediction_model.linear_weights,2);
    elseif (isfield(prediction_model,'param_weights'))
        num_forces = size(prediction_model.param_weights,2);
    elseif (isfield(prediction_model,'cutoff_index'))
        num_forces = length(prediction_model.mean_Y);
    else
        % No model present
        forces_training = zeros(num_images, 3);
        use_image = false(num_images,1);
        return;
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
    
    % Preallocate outputs
    forces_training = zeros(num_images, num_forces);
    use_image = false(num_images,1);
    for imageIdx = 1:num_images
        % User-friendly message
        if ((mod(imageIdx,50) == 0) && (verbose))
            if (debug_mode)
                fprintf('\tProcessing image %04d\n', imageIdx);
            else
                fprintf('.');
            end
        end
        
        % Extract current image
        current_image = pixels(imageIdx, :);
        
        % Predict force on this image
        [forces_training(imageIdx,:), use_image(imageIdx)] = predict_force(current_image, prediction_model, options);
    end % imageIdx

end % predict_forces_on_datset