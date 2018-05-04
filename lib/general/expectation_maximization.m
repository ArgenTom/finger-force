%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Perform Expectation Maximization given a matrix of data values, a mask
%   showing the initial region estimates for each data value, and a
%   structure of options
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [mask_vector, voting_weight, valid_result] = expectation_maximization(pixel_matrix, mask_vector, options)

    % Process options
    if (nargin() == 2)
        [err_tol, max_iter, max_changes, verbose, debug_mode] = process_options([]);
    elseif (nargin() == 3)
        [err_tol, max_iter, max_changes, verbose, debug_mode] = process_options(options);
    else
        error('Must have 2 or 3 inputs!');
    end
    
    % Determine size of inputs
    [num_pixels, num_variables] = size(pixel_matrix);
    
    % Determine desired number of regions
    num_regions = max(mask_vector) - min(mask_vector) + 1;
    if (max(mask_vector) ~= num_regions)
        error('Mask vector is inconsistent!');
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Calculate the initial estimates for each region's properties, which
    %   are as follows:
    %       Weight (alpha), initialized to 1/(number of regions)
    %       Mean vector (x_bar), initialized to average of the pixel
    %           values
    %       Covariance matrix (Sigma), initialized to the identity
    %           matrix
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    region_weights = ones(num_regions, 1) / num_regions;
    region_means = zeros(num_regions, num_variables);
    region_covariances = zeros(num_variables, num_variables, num_regions);
    
    % Process data for each region
    for regionIdx = 1:num_regions
        % Find all pixels pertaining to this region
        region_pixels = find(mask_vector == regionIdx);
        num_rp = length(region_pixels);
        
        % Find the mean of the pixel values of the current region
        region_means(regionIdx, :) = sum(pixel_matrix(region_pixels, :))/ num_rp;
        
        % Initialize the Covariance Matrix
        region_covariances(:,:,regionIdx) = eye(num_variables);
    end % regionIdx
    
    % Calculate the PDF for each pixel/region combination
    pdf = calculate_pdf(pixel_matrix, region_means, region_covariances);
    
    % Calculate the probability that each Gaussian fits each pixel
    prob_fit = calculate_fit_probability(region_weights, pdf);
    
    % Calculate the Log Likelihood that the current estimates are correct
    last_likelihood = 0;
    
    % Initialize iteration parameters
    iter = 1;
    pct_err = 100;
    changed_iter = 1;
    old_mask = mask_vector;
    
    % Display the region properties
    if (verbose)
        fprintf('Iter %03d | Likelihood %9.5e | Pct Err %7.3f | Changed %d\n', iter, last_likelihood, pct_err, changed_iter);
        if (debug_mode)
            fig_debug = figure();
            plot(pixel_matrix(:,1), mask_vector, 'r.');
            hold on;
            plot(pixel_matrix(:,2), mask_vector, 'g.');
            plot(pixel_matrix(:,3), mask_vector, 'b.');
            hold off;
        end
    end
    
    % Iterate as long as:
    %   (1) Maximum iteration count has not been reached.
    %   (2) Error criteria has not been reached.
    %   (3) The region mask has changed within a reasonable number of
    %       iterations.
    while ((iter < max_iter) && (pct_err > err_tol) && (iter - changed_iter < max_changes))
        % Increment the iteration counter
        iter = iter + 1;
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Update the values of each parameter
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        for regionIdx = 1:num_regions
            % Update the weight parameters
            prob_sum = sum(prob_fit(regionIdx,:));
            region_weights(regionIdx) = prob_sum / num_pixels;
            
            % Update the mean parameters
            num_mean = zeros(1,num_variables);
            for pixelIdx = 1:num_pixels
                x = pixel_matrix(pixelIdx,:);
                num_mean = num_mean + x * prob_fit(regionIdx,pixelIdx);
            end % pixelIdx
            region_means(regionIdx,:) = num_mean / prob_sum;
            
            % Update the covariance matrices
            num_cov = zeros(num_variables);
            for pixelIdx = 1:num_pixels
                x = pixel_matrix(pixelIdx,:);
                num_cov = num_cov + prob_fit(regionIdx,pixelIdx) * (x - region_means(regionIdx,:))' * (x - region_means(regionIdx,:));
            end % pixelIdx
            region_covariances(:, :, regionIdx) = num_cov / sum(prob_fit(regionIdx,:));
        end % regionIdx
        
        % Verify that the covariance matrices are not NaN
        if (any(isnan(region_covariances(:))))
            keyboard;
        end
        
        % Calculate the PDF for each pixel/region combination
        pdf = calculate_pdf(pixel_matrix, region_means, region_covariances);
        
        % Calculate the probability that each Gaussian fits each pixel
        prob_fit = calculate_fit_probability(region_weights, pdf);
        
        % Calculate the Log Likelihood (i.e., the termination criteria).
        new_likelihood = calculate_log_likelihood(region_weights, prob_fit);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Recalculate the percent error and update the Log Likelihood
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        pct_err = abs((new_likelihood - last_likelihood) / new_likelihood) * 100;
        last_likelihood = new_likelihood;
        
        % Update the current mask vector
        mask_vector = zeros(num_pixels, 1);
        voting_weight = zeros(num_pixels, 1);
        for pixelIdx = 1:num_pixels
            % Determine the region to which this pixel belongs
            voting_weight(pixelIdx) = max(prob_fit(:,pixelIdx));
            regionIdx = find(prob_fit(:,pixelIdx) == voting_weight(pixelIdx));
            if (length(regionIdx) ~= 1)
                keyboard;
            end
            
            % Populate the pixel with the appropriate color
            mask_vector(pixelIdx) = regionIdx;
        end % pixelIdx
        
        % Determine whether the image mask has changed
        mask_error = mask_vector - old_mask;
        if (any(mask_error(:) ~= 0))
            changed_iter = iter;
        end
        old_mask = mask_vector;
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Graphical/Debugging outputs
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        if (verbose)
            % Display the region properties
            fprintf('Iter %03d | Likelihood %9.5e | Pct Err ', iter, new_likelihood);
            if (pct_err >= 1000)
                fprintf('(Large)');
            else
                fprintf('%7.3f', pct_err);
            end
            fprintf(' | Changed %d\n', changed_iter);
            
            % Reshape the image
            if (isfield(options,'image_size'))
                image_size = options.image_size;
                imgMask = reshape(mask_vector, image_size(1), image_size(2));
                imshow(imgMask,[]);
                drawnow();
            end
        end
        
    end
    
    % Print an "end of process" line to indicate that we're moving on
    if (verbose)
        fprintf('=======================\n');
        
        % Display the reason the iteration ended
        if (iter >= max_iter)
            fprintf('\t(1) Reached maximum # of iterations\n');
        end
        if (pct_err <= err_tol)
            fprintf('\t(2) Reached minimum desired percent error\n');
        end
        if (iter - changed_iter >= max_changes)
            fprintf('\t(3) Image has not changed!\n');
        end
    end
    
    % Declare validity of result based on percent error
    if (isnan(pct_err))
        valid_result = false;
    else
        valid_result = true;
    end

end % em

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Process the option structure to extract all expected options that will be
% used by this function.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [err_tol, max_iter, max_changes, verbose, debug_mode] = process_options(options)

    if (isfield(options,'err_tol'))
        err_tol = options.err_tol;
    else
        err_tol = 0.01;
    end
    if (isfield(options,'max_iter'))
        max_iter = options.max_iter;
    else
        max_iter = 20;
    end
    if (isfield(options,'max_changes'))
        max_changes = options.max_changes;
    else
        max_changes = Inf;
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculate the PDF for each pixel/region combination
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function pdf = calculate_pdf(pixel_matrix, region_means, region_covariances)

    % Determine size of inputs
    [num_pixels, num_variables] = size(pixel_matrix);
    [num_regions, num_variables2] = size(region_means);
    if ((num_variables ~= num_variables2) || (size(region_covariances,3) ~= num_regions) || (size(region_covariances,1) ~= num_variables) || (size(region_covariances,1) ~= size(region_covariances,2)))
        error('Input sizes do not match!');
    end
    
    % Initialize output
    pdf = zeros(num_regions, num_pixels);
    
    % Process each region
    for regionIdx = 1:num_regions
        % Calculate the denominator of the PDF
        denominator = (2*pi)^(0.5*num_variables) * sqrt(det(region_covariances(:,:,regionIdx)));
        
        % Determine whether the covariance matrix is invertible
        for pixelIdx = 1:num_pixels
            % Extract the current pixel
            x = pixel_matrix(pixelIdx,:);
            
            % Calculate the probability density function
            numerator = exp(-0.5*(x - region_means(regionIdx,:))*inv(region_covariances(:,:,regionIdx))*(x - region_means(regionIdx,:))');
            pdf(regionIdx, pixelIdx) = numerator / denominator;
        end % pixelIdx
    end % regionIdx

end % calculate_pdf

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculate the probability that each Gaussian fits each pixel
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function prob_fit = calculate_fit_probability(region_weights, pdf)

    % Determine size of inputs
    [num_regions, num_pixels] = size(pdf);
    if (length(region_weights) ~= num_regions)
        error('Input sizes do not match!');
    end
    
    % Initialize output
    prob_fit = zeros(num_regions, num_pixels);
    
    % Calculate the sum belonging to the denominator for each pixel
    denominator = zeros(1,num_pixels);
    for regionIdx = 1:num_regions
        denominator = denominator + region_weights(regionIdx) * pdf(regionIdx, :);
    end % regionIdx
    
    % Calculate the probability that each Gaussian fits each pixel
    for regionIdx = 1:num_regions
        prob_fit(regionIdx, :) = region_weights(regionIdx) * pdf(regionIdx, :) ./ denominator;
    end % regionIdx

end % calculate_fit_probability

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculate the Log Likelihood (i.e., the termination criteria).  When
% this increases by less than 1% from one iteration to the next, the
% loop terminates.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function log_likelihood = calculate_log_likelihood(region_weights, prob_fit)

    % Determine size of inputs
    [num_regions, num_pixels] = size(prob_fit);
    if (length(region_weights) ~= num_regions)
        error('Input sizes do not match!');
    end
    
    % Calculate the overall probability function
    overall_prob = zeros(1,num_pixels);
    for regionIdx = 1:num_regions
        overall_prob = overall_prob + region_weights(regionIdx) * prob_fit(regionIdx,:);
    end % regionIdx

    % Calculate the Log Likelihood
    log_likelihood = sum(log10(overall_prob));

end % calculate_log_likelihood
