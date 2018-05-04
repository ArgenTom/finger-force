% FIND_LWLR_VALUE Locally-Weighted Linear Regression
% 

function [pixel_fit, pixel_coefficients] = find_lwlr_fit(force, pixel, options)

    % Verify input parameters
    if (nargin == 2)
        [kernel_width, debug_mode] = unpack_options([]);
    elseif (nargin() == 3)
        [kernel_width, debug_mode] = unpack_options(options);
    else
        error('Must have 2 or 3 inputs!\n');
    end
    
    % Verify size of inputs
    [nImages, nXvars] = size(force);
    nPixels = length(pixel);
    nCoeffs = nXvars+1;
    if (nPixels ~= nImages)
        error('Force and Pixel data are inconsistent!\n');
    end
    
    % Preallocate output
    pixel_fit = zeros(nImages, 1);
    pixel_coefficients = zeros(nImages, nCoeffs);
    maxNearPts = 0;
    maxImageIdx = 0;
    
    % Iterate to calculate outputs
    for imageIdx=1:nImages,
        if ((debug_mode) && (mod(imageIdx,500) == 0))
            fprintf('\tProcessing Image %04d\n', imageIdx);
        end
        % Select a calculation point
        xImage = force(imageIdx,:);
        
        % Determine distances to every force point
        distances = sqrt(sum((force - ones(nImages,1)*xImage).^2,2));
        
        % Only iterate through points within 3 kernel width
        nearPoints = find(distances <= 3*kernel_width);
        nNearPts = length(nearPoints);
        if (nNearPts > maxNearPts)
            maxNearPts = nNearPts;
            maxImageIdx = imageIdx;
        end
        
        % Preallocate the X and Y arrays
        X_matrix = zeros(nNearPts, nCoeffs);
        Y_vector = zeros(nNearPts, 1);
        
        % Iterate through nearby points to populate the arrays
        for nearPointIdx=1:nNearPts,
            % Select an (x,y) point
            xNear = force(nearPoints(nearPointIdx),:);
            yNear = pixel(nearPoints(nearPointIdx));
            point_distance = distances(nearPoints(nearPointIdx));
            
            % Determine the weighting factor w_Fixed
            wNear = exp(-point_distance^2 / (2*kernel_width^2));
            
            % Populate the X_matrix
            X_matrix(nearPointIdx,:) = [xNear 1]*wNear;
            
            % Populate the Y_vector
            Y_vector(nearPointIdx) = wNear * yNear;
        end
        
        % Calculate the coefficients
        coefficient_matrix = (X_matrix \ Y_vector)';
        
        % Assemble the output
        pixel_fit(imageIdx) = [xImage 1]*coefficient_matrix';
        pixel_coefficients(imageIdx,:) = coefficient_matrix;
    end
    if (debug_mode)
        fprintf('\t\t%03d | %d/%d\n', maxImageIdx, maxNearPts, nImages);
    end

end % function

function [kernel_width, debug_mode] = unpack_options(options)

    if (isfield(options,'kernel_width'))
        kernel_width = options.kernel_width;
    else
        kernel_width = 1;
    end
    if (isfield(options,'debug_mode'))
        debug_mode = options.debug_mode;
    else
        debug_mode = false;
    end

end % unpack_options
