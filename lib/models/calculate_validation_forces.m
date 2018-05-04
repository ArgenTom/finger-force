% Calculate the predicted forces using the "validation" method.  In other
% words, split the force and image data into two sets.  Use one set to
% calibrate the model (the Training Data) and use the model to predict the
% forces on the other set (the Validation Data).

function [validation_sig, sig_valid, validation_eig, validation_shp, validation_tex, validation_app] = calculate_validation_forces(pixel_matrix, force_matrix, shape_matrix, gray_matrix, model_matrix, validation_tests, options)

    % Verify the input parameters
    if (nargin() == 6)
        [run_sigmoid, verbose, debug_mode] = process_options([]);
    elseif (nargin() == 7)
        [run_sigmoid, verbose, debug_mode] = process_options(options);
    else
        error('Must have 4 inputs!');
    end
    
    % Determine size of inputs
    [num_files, num_pixels] = size(pixel_matrix);
    num_forces = size(force_matrix, 2);
    num_shape = size(shape_matrix, 2);
    num_tex = size(gray_matrix, 2);
    num_app = size(model_matrix, 2);
    if ((size(force_matrix,1) ~= num_files) || (size(shape_matrix,1) ~= num_files) || (size(gray_matrix,1) ~= num_files) || (size(model_matrix,1) ~= num_files))
        error('Inputs are not of compatible size!');
    end
    [num_validation, num_tests] = size(validation_tests);
    
    % Preallocate outputs
    validation_sig = zeros(num_validation, num_tests, num_forces);
    sig_valid = zeros(num_validation, num_tests);
    validation_eig = zeros(num_validation, num_tests, num_forces);
    validation_shp = zeros(num_validation, num_tests, num_forces);
    validation_tex = zeros(num_validation, num_tests, num_forces);
    validation_app = zeros(num_validation, num_tests, num_forces);
    if (debug_mode)
        t = zeros(num_tests,3);
    end
    
    % Populate outputs
    for testIdx = 1:num_tests
        if (verbose)
            fprintf('Performing Test #%02d', testIdx);
        end
        
        % Generate the validation pixels, gray-level parameters and forces
        validation_images = validation_tests(:,testIdx);
        validation_pixels = pixel_matrix(validation_images,:);
        validation_shape = shape_matrix(validation_images,:);
        validation_texture = gray_matrix(validation_images,:);
        validation_appearance = model_matrix(validation_images,:);
        validation_forces = force_matrix(validation_images,:);
        
        % Generate the training pixels, gray-level parameters and forces
        training_images = 1:num_files;
        training_images(validation_images) = [];
        training_pixels = pixel_matrix(training_images,:);
        training_shape = shape_matrix(training_images,:);
        training_texture = gray_matrix(training_images,:);
        training_appearance = model_matrix(training_images,:);
        training_forces = force_matrix(training_images,:);
        
        % Create the models using the training images
        if (run_sigmoid)
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
        end
        if (debug_mode)
            tic;
        end
        eig_model = form_eigennail_model(training_forces, training_pixels);
        if (debug_mode)
            t(testIdx,2) = toc;
        end
        if (verbose)
            fprintf('...Eig');
        end
        if (debug_mode)
            tic;
        end
        [shape_model, texture_model, appearance_model] = form_AAMFinger_model(training_forces, training_shape, training_texture, training_appearance);
        if (debug_mode)
            t(testIdx,3) = toc;
        end
        if (verbose)
            fprintf('...AAM');
        end
        
        % Check the validation images using the prediction models
        if (run_sigmoid)
            [validation_sig(:, testIdx, :), sig_valid(:,testIdx)] = predict_forces_on_dataset(validation_pixels, sig_model);
        end
        [validation_eig(:, testIdx, :), use_images] = predict_forces_on_dataset(validation_pixels, eig_model);
        [validation_shp(:, testIdx, :), use_images] = predict_forces_on_dataset(validation_shape, shape_model);
        [validation_tex(:, testIdx, :), use_images] = predict_forces_on_dataset(validation_texture, texture_model);
        [validation_app(:, testIdx, :), use_images] = predict_forces_on_dataset(validation_appearance, appearance_model);
        if (verbose)
            fprintf('...Predicted!\n');
        end
    end % testIdx
    if (debug_mode)
        mean_times = mean(t);
        fprintf('Mean Model Formulation Times:\n');
        fprintf('\tLinearized Sigmoid:   %5.2f\n', mean_times(1));
        fprintf('\tEigenNail Magnitude:  %5.2f\n', mean_times(2));
        fprintf('\tAAM Parameters (all): %5.2f\n', mean_times(3));
    end

end % calculate_validation_forces

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
