% FORM_SIGMOID_MODEL Calibrates a Linearized Sigmoid Model
% given a matrix of pixel information (pixel_matrix) and a matrix of force
% information (force_matrix).  For each pixel p_i, the model has the form:
%
%   p_i = a_i + b_i*f_x + c_i*f_y + d_i*f_z
% 
% where [f_x f_y f_z] are the forces on the fingerpad in the image, and
% [a_i b_i c_i d_i] are the parameters of the model.
% 
% Written by Thomas R. Grieve
% 20 March 2012
% University of Utah
%
% 

function prediction_model = form_sigmoid_model(forces, pixels, options)

    % Verify inputs
    if (nargin() == 2)
        options = [];
    elseif (nargin() ~= 3)
        error('Must have 2 or 3 inputs!');
    end
    [min_saturation, max_gradient_fraction, min_corr_coeff, mask_image, verbose, debug_mode] = unpack_options(options);
    
    % Verify input sizes
    [num_images, num_pixels] = size(pixels);
    [num_images2, num_forces] = size(forces);
    if (num_images ~= num_images2)
        error('Pixel Matrix is not consistent with Force Matrix!');
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Begin "form_prediction_model"
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Preallocate the output
    coefficients = zeros(num_pixels,4);
    saturation = zeros(num_pixels,2);
    use_pixel = true(num_pixels,1);
    covariance_matrix = zeros(num_pixels,1);
    options.debug_mode = false;
    
    % Iterate through pixels
    for pixelIdx = 1:num_pixels
        % Print a user-friendly message
        if ((verbose) && (mod(pixelIdx,round(num_pixels/10)) == 0))
            if (debug_mode)
%                fprintf('\tProcessing pixel %04d / %04d (%5.1f%%)\n', pixelIdx, num_pixels, (pixelIdx/num_pixels)*100);
            else
                fprintf('.');
            end
        end
        
        % Calculate LWLR fit
        [pixel_fit, pixel_coefficients] = find_lwlr_fit(forces,pixels(:,pixelIdx),options);
        
        % Calculate gradient of the pixel data
        grad_mag = sqrt(sum((pixel_coefficients(:,1:num_forces)).^2,2));
        
        % Determine images to keep
        maximum_gradient = max(grad_mag);
        keep_grad = (grad_mag > max_gradient_fraction*maximum_gradient);
        num_to_keep = sum(keep_grad);
        if (num_to_keep < num_forces+1)
            use_pixel(pixelIdx) = false;
        end
        
        % Determine multiple least-squares coefficient vector for kept data
        force_data = [forces(keep_grad,:) ones(num_to_keep,1)];
        pixel_data = pixels(keep_grad, pixelIdx);
        if (rank(force_data) < num_forces+1)
            use_pixel(pixelIdx) = false;
            continue;
        end
        coefficient_vector = force_data \ pixel_data;
        
        % Determine goodness of fit using the correlation coefficient
        pixel_fit = force_data*coefficient_vector;
        y_bar = sum(pixel_data) / num_to_keep;
        S_t = sum((pixel_data - y_bar).^2);
        S_r = sum((pixel_data - pixel_fit).^2);
        correlation_coefficient = sqrt((S_t - S_r)/S_t);
        if (correlation_coefficient < min_corr_coeff)
            use_pixel(pixelIdx) = false;
        end
        
        % Determine saturation limits and verify that the pixel varies enough
        sat_lower = min(pixel_data);
        sat_upper = max(pixel_data);
        if (sat_upper - sat_lower < min_saturation)
            use_pixel(pixelIdx) = false;
        end
        
        % Store coefficients and saturation limits
        coefficients(pixelIdx,:) = coefficient_vector;
        saturation(pixelIdx,:) = [sat_lower sat_upper];
        covariance_matrix(pixelIdx) = var(pixels(:,pixelIdx));
        
        % Plot the data, as desired
        if (debug_mode)
            if (use_pixel(pixelIdx))
                fprintf('Pixel %04d | Good\n', pixelIdx);
            else
                continue;
            end
            if (exist('fig_plot','var'))
                figure(fig_plot);
            else
                fig_plot = figure;
            end
            forceIdx = mod(pixelIdx-1,3)+1;
            plot(forces(:,forceIdx), pixels(:,pixelIdx), 'k.');
            hold on;
            
            plot(forces(~keep_grad,forceIdx), pixels(~keep_grad,pixelIdx), 'ro');
            plot(force_data(:,forceIdx), pixel_fit, 'g+');
            hold off;
        end
    end % pixelIdx
    
    % Form prediction model structure
    prediction_model.slope_matrix = coefficients(use_pixel,1:num_forces);
    prediction_model.pixel_offset = coefficients(use_pixel,end);
    
    % Calculate covariance matrix (valid pixels only)
    prediction_model.covariance_matrix = diag(covariance_matrix(use_pixel));
    
    % Finish prediction model structure
    prediction_model.saturation = saturation(use_pixel,:);
    prediction_model.use_pixel = use_pixel;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % End "form_prediction_model"
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
%     % Display the coefficients as images, if desired
%     if (debug_mode)
%         fig_coeffs = display_sigmoid_model(prediction_model, mask_image, options);
%     end

end % form_sigmoid_model

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Unpack the options structure
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [min_saturation, max_gradient_fraction, min_corr_coeff, mask_image, verbose, debug_mode] = unpack_options(options)

    % Assign default values if the fields do not exist
    if (isfield(options,'min_saturation'))
        min_saturation = options.min_saturation;
    else
        min_saturation = 5/255;
    end
    if (isfield(options,'max_gradient_fraction'))
        max_gradient_fraction = options.max_gradient_fraction;
    else
        max_gradient_fraction = 0.2;
    end
    if (isfield(options,'min_corr_coeff'))
        min_corr_coeff = options.min_corr_coeff;
    else
        min_corr_coeff = 0.8;
    end
    
    % Providing verbose output (i.e., how many pixels have been processed)
    if (isfield(options,'verbose'))
        verbose = options.verbose;
    else
        verbose = false;
    end
    
    % Display debugging information (i.e., display coefficient images)
    if (isfield(options,'debug_mode'))
        debug_mode = options.debug_mode;
        
        % Must have an image mask if we are to enter debug mode
        if (debug_mode)
            if (isfield(options,'mask_image'))
                mask_image = options.mask_image;
            else
                error('Cannot enter debug mode without a mask_image!');
            end
        else
            mask_image = [];
        end
    else
        debug_mode = false;
        mask_image = [];
    end

end % unpack_options
