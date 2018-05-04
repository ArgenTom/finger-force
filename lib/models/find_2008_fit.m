% Nonlinear least-squares fit to a sigmoid function
% 

function [pixel_estimated, pixel_coefficients, fit_data] = find_2008_fit(forces,pixels,options)

    % Process optional input
    if (nargin() == 2)
        [max_iter, err_tol, use_damped, alpha_value, num_interp, debug_mode, default_position] = process_options([]);
    elseif (nargin() == 3)
        [max_iter, err_tol, use_damped, alpha_value, num_interp, debug_mode, default_position] = process_options(options);
    else
        error('Must have 2 or 3 inputs!');
    end
    
    % Determine size of inputs
    num_points = length(pixels);
    
    % Initial parameter estimates
    parameter_vector = get_initial_parameter_estimates(forces, pixels);
    num_params = length(parameter_vector);
    
    % Set up iteration parameters
    iter = 0;
    res = 100;
    alpha_current = alpha_value;
    
    if (debug_mode)
        parameter_array = zeros(max_iter, num_params);
        obs_index = zeros(1,5);
    end
    
    % Iterate
    while ((iter < max_iter) && (res > err_tol))
        % Increment iteration counter
        iter = iter + 1;
        
        % Populate calibration matrix and estimated pixel vector
        calib_matrix = populate_matrix(parameter_vector, forces);
        pixel_est = populate_estimate(parameter_vector, forces);
        if (debug_mode)
            try
                fprintf('Iter %02d | Rank %d', iter, rank(calib_matrix));
            catch exception
                keyboard;
            end
        end
        
        % Task variable scaling
        task_weight = std(pixels);
        pixel_measured = pixels / task_weight;
        pixel_estimated = pixel_est / task_weight;
        calib_matrix = calib_matrix / task_weight;
        
        % Parameter scaling
        param_weights = zeros(num_params,1);
        for paramIdx = 1:num_params
            column_norm = norm(calib_matrix(:,paramIdx));
            if (column_norm == 0)
                param_weights(paramIdx) = 1;
            else
                param_weights(paramIdx) = column_norm;
            end
            
            % Scale calibration matrix
            calib_matrix(:,paramIdx) = calib_matrix(:,paramIdx) / param_weights(paramIdx);
        end % paramIdx
        
        % Calculate the update vector
        dy = pixel_measured - pixel_estimated;
        if (use_damped)
            delta_phi = (calib_matrix' * calib_matrix + lambda^2 * eye(num_params)) \ calib_matrix' * dy;
        else
            delta_phi = calib_matrix \ dy;
        end
        
        % Update the error of the parameter vector
        if (res > norm(delta_phi))
            alpha_current = alpha_current * 1.1;
        else
            alpha_current = alpha_current * 0.5;
        end
        res = norm(delta_phi);
        
        % Undo parameter scaling
        delta_phi = diag(param_weights) \ delta_phi;
        
        % Update the parameter vector
        parameter_new = parameter_vector + alpha_current*delta_phi;
        
        if (debug_mode)
            fprintf(' | Alpha %5.3f | Resid %6.3f\n', alpha_current, res);
            % Plot data
            if (exist('fig_main','var'))
                figure(fig_main);
            else
                fig_main = figure();
                set(fig_main,'Position',default_position);
                ax_original = subplot(2,2,1);
                ax_scaled = subplot(2,2,2);
                ax_fit = subplot(2,2,3);
                ax_fit_sc = subplot(2,2,4);
            end
            plot(ax_original, pixels, pixel_est, 'k.');
            axis(ax_original,'equal');
            plot(ax_scaled, pixel_measured, pixel_estimated, 'k.');
            axis(ax_scaled,'equal');
            plot(ax_fit, forces, pixels, 'k.', forces, pixel_est, 'r.');
            plot(ax_fit_sc, forces, pixel_measured, 'k.', forces, pixel_estimated, 'r.');
            
            % Perform observability/identifiability calculations
            [U,S,V] = svd(calib_matrix,'econ');
            eigenvalues = diag(S);
            obs_index(1) = prod(eigenvalues)^(1/num_params) / sqrt(num_points);
            obs_index(2) = eigenvalues(end) / eigenvalues(1);
            obs_index(3) = eigenvalues(end);
            obs_index(4) = eigenvalues(end)^2 / eigenvalues(1);
            obs_index(5) = sum(1 ./ eigenvalues.^2);
            ident_ind = eigenvalues(1) / eigenvalues(end);
            parameter_array(iter+1,:) = parameter_new;
        end
        
        % Update the parameter vector
        parameter_vector = parameter_new;
    end
    
    % Calculate outputs
    [pixel_estimated, pixel_coefficients, fit_data] = populate_estimate(parameter_vector, forces, num_interp);

end % find_2008_fit

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Estimate initial values for the parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function parameter_vector = get_initial_parameter_estimates(forces, pixels)

    % Initial values for the lower asymptote (a) and response range (b)
    a = floor(min(pixels));
    b = ceil(max(pixels)) - a;
    
    % Initial value for the force offset (f0)
    f0 = mean(forces);
    
    % Initial value for the relative response rate (c)
    c = (forces - f0) \ log(b ./ (pixels - a) - 1);
    
    % Assemble output vector
    parameter_vector = [a; b; c; f0];

end % get_initial_parameter_estimates

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Populate calibration matrix
%   Model is p = a + b / (1 + exp(exponent_value)) where
%           exponent_value = c*f + d
% 
%       Parameter vector is [a b c d]
%       Forces matrix is [fx fy fz] where each column is a vector
%       Calibration matrix column for parameter 'a' is identically 1.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function calib_matrix = populate_matrix(parameter_vector, forces)

    % Determine size of inputs
    num_params = length(parameter_vector);
    num_points = size(forces, 1);
    
    % Extract parameters
    b = parameter_vector(2);
    c = parameter_vector(3);
    d = parameter_vector(4);
    
    % Preallocate outputs
    calib_matrix = ones(num_points, num_params);
    
    % Calculate exponent value
    exponent_value = c*(forces - d);
    
    % Populate calibration matrix for b
    calib_matrix(:, 2) = 1 ./ (1 + exp(exponent_value));
    
    % Populate calibration matrix for c
    calib_matrix(:, 3) = -b*exp(exponent_value).*forces ./ (1 + exp(exponent_value)).^2;
    
    % Populate calibration matrix for d
    calib_matrix(:, 4) = b*exp(exponent_value)*c ./ (1 + exp(exponent_value)).^2;

end % populate_matrix

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Populate estimate of output (pixel intensity value)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [pixel_estimated, pixel_coefficients, fit_data] = populate_estimate(parameter_vector, forces, num_interp)

    % Determine size of inputs
    num_points = size(forces, 1);
    
    % Extract parameters
    a = parameter_vector(1);
    b = parameter_vector(2);
    c = parameter_vector(3);
    d = parameter_vector(4);
    
    % Calculate exponent value
    exponent_value = c*(forces - d);
    
    % Calculate estimated output value
    pixel_estimated = a + b ./ (1 + exp(exponent_value));
    
    % Calculate pixel coefficients, if needed
    if (nargout() >= 2)
        pixel_coefficients = b*exp(exponent_value)*c ./ (1 + exp(exponent_value)).^2;
    end
    
    % Calculate interpolated (fitting) forces, if needed
    if (nargout() == 3)
        if (nargin() ~= 3)
            error('Must have 3 inputs!');
        end
        force_out = (0:num_interp)*(max(forces) - min(forces)) / num_interp + min(forces);
        exp_out = c*(force_out - d);
        pixel_out = a + b ./ (1 + exp(exp_out));
        fit_data.f = force_out;
        fit_data.p = pixel_out;
    end

end % populate_estimate

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Process input options
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [max_iter, err_tol, use_damped, alpha_value, num_interp, debug_mode, default_position] = process_options(options)

    % Maximum number of iterations
    if (isfield(options,'max_iter'))
        max_iter = options.max_iter;
    else
        max_iter = 50;
    end
    
    % Minimum error value before terminating
    if (isfield(options,'err_tol'))
        err_tol = options.err_tol;
    else
        err_tol = 0.01;
    end
    
    % Use Damped Least Squares
    if (isfield(options,'use_damped'))
        use_damped = options.use_damped;
    else
        use_damped = false;
    end
    
    % Weight given to new estimate
    if (isfield(options,'alpha_value'))
        alpha_value = options.alpha_value;
    else
        alpha_value = 0.1;
    end
    
    % Number of interpolated points to use for fitted points
    if (isfield(options,'num_interp'))
        num_interp = options.num_interp;
    else
        num_interp = 200;
    end
    
    % Enter Debugging Mode
    if (isfield(options,'debug_mode'))
        debug_mode = options.debug_mode;
    else
        debug_mode = false;
    end
    
    % Set default position for plot windows
    if (isfield(options,'default_position'))
        default_position = options.default_position;
    else
        default_position = [45 45 1400 400];
    end

end % process_options
