% Calculate the predicted forces using the "validation" method, only for
% the Linearized Sigmoid method.  This is used to 'make up' for not running
% the LinSig model on certain data sets originally.
% 
% This function splits the force and image data into two sets.  Use one set
% to calibrate the model (the Training Data) and use the model to predict
% the forces on the other set (the Validation Data).
% 

function [validation_sig, sig_valid] = calculate_validation_sigmoid(pixel_matrix, force_matrix, validation_tests, options)

    % Verify the input parameters
    if (nargin() == 3)
        [run_sigmoid, verbose, debug_mode] = process_options([]);
    elseif (nargin() == 4)
        [run_sigmoid, verbose, debug_mode] = process_options(options);
    else
        error('Must have 3 or 4 inputs!');
    end
    
    % Determine size of inputs
    [num_files, num_pixels] = size(pixel_matrix);
    num_forces = size(force_matrix, 2);
    if (size(force_matrix,1) ~= num_files)
        error('Inputs are not of compatible size!');
    end
    [num_validation, num_tests] = size(validation_tests);
    
    % Preallocate outputs
    validation_sig = zeros(num_validation, num_tests, num_forces);
    sig_valid = zeros(num_validation, num_tests);
    if (debug_mode)
        t = zeros(num_tests,1);
    end
    
    % Populate outputs
    for testIdx = 1:num_tests
        if (verbose)
            fprintf('Performing Test #%02d', testIdx);
        end
        
        % Generate the validation pixels, gray-level parameters and forces
        validation_images = validation_tests(:,testIdx);
        validation_pixels = pixel_matrix(validation_images,:);
        validation_forces = force_matrix(validation_images,:);
        
        % Generate the training pixels, gray-level parameters and forces
        training_images = 1:num_files;
        training_images(validation_images) = [];
        training_pixels = pixel_matrix(training_images,:);
        training_forces = force_matrix(training_images,:);
        
        % Create the models using the training images
        if (debug_mode)
            tic;
        end
        sig_model = form_sigmoid_model(training_forces, training_pixels);
        if (debug_mode)
            t(testIdx,1) = toc;
        end
        if (verbose)
            fprintf('...Sig');
        end
        
        % Check the validation images using the prediction models
        [validation_sig(:, testIdx, :), sig_valid(:,testIdx)] = predict_forces_on_dataset(validation_pixels, sig_model);
        if (verbose)
            fprintf('...Predicted!\n');
        end
    end % testIdx
    if (debug_mode)
        mean_times = mean(t);
        fprintf('Mean Model Formulation Times:\n');
        fprintf('\tLinearized Sigmoid:   %5.2f\n', mean_times);
    end

end % calculate_validation_sigmoid

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Process the input options and assign default values as needed
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [run_sigmoid, verbose, debug_mode] = process_options(options)

    if (isfield(options,'run_sigmoid'))
        run_sigmoid = options.run_sigmoid;
    else
        run_sigmoid = false;
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

end % process_options
