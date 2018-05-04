% MATCH_HISTOGRAM 2D Image Histogram Matching
%
% imgOut = match_histogram(imgIn, freq, bins) performs histogram
%          matching on imgIn to create a new image imgOut whose histogram
%          most nearly approximates that specified by the inputs (freq,
%          bins).
% 
% imgIn        = 2D image to process
% desired_hist = Desired histogram frequencies, or heights of the PDF:
%                length(freq)=n
% bins         = Desired histogram bin edges: length(bins)=n+1
% imgOut       = Output image
% 
% Thomas R. Grieve
% for CS 6640 - Image Processing
% Fall 2009
% University of Utah
% Created: 23 September 2009
% Updated:  1 October 2012

function imgOut = match_histogram(imgIn, desired_hist, bins, options)

    % Process inputs
    if (nargin() == 3)
        [debug_mode, verbose, figure_ID] = process_options([]);
    elseif (nargin() == 4)
        [debug_mode, verbose, figure_ID] = process_options(options);
    else
        error('Must have 2 or 3 inputs!');
    end
    
    % Initialize output
    imgOut = imgIn;
    num_bins = length(desired_hist);
    num_pixels = numel(imgIn);
    
    % Calculate the histogram of the current image
    current_hist = zeros(num_bins, 1);
    for binIdx = 1:num_bins
        temp = ((imgIn > bins(binIdx)) & (imgIn <= bins(binIdx+1)));
        current_hist(binIdx) = sum(temp(:)) / num_pixels;
    end % binIdx
    
    % Calculate the CDF of both histograms
    desired_cdf = round((cumsum(desired_hist) / sum(desired_hist)) * 255);
    current_cdf = round((cumsum(current_hist) / sum(current_hist)) * 255);
    
    % Match the two histograms
    for binIdx = 1:num_bins
        % Determine the pixels in the current image that fall in this bin
        image_pixels = ((imgIn > bins(binIdx)) & (imgIn <= bins(binIdx+1)));
        
        % Find the matching bin of the desired CDF
        matching_bin = find(desired_cdf <= current_cdf(binIdx), 1, 'last');
        
        % Assign the output value to be the center of the matching bin
        bin_center = (bins(matching_bin) + bins(matching_bin+1)) / 2;
        imgOut(image_pixels) = bin_center;
    end % binIdx
    
    if (debug_mode)
        bin_centers = (bins(2:end) + bins(1:end-1))/2;
        max_freq = ceil(max([desired_hist(:); current_hist(:)])*500)/500;
        figure(figure_ID);
        subplot(2,1,1);
        plot(bin_centers, desired_hist, 'k-', bin_centers, current_hist, 'r-');
        axis([bins(1) bins(end) 0 max_freq]);
        legend('Desired','Current');
        title('PDF');
        subplot(2,1,2);
        plot(bin_centers, desired_cdf/255, 'k-', bin_centers, current_cdf/255, 'r-');
        legend('Desired','Current');
        title('CDF');
    end

end % equalize_histogram

function [debug_mode, verbose, figure_ID] = process_options(options)

    if (isfield(options,'debug_mode'))
        debug_mode = options.debug_mode;
    else
        debug_mode = false;
    end
    if (isfield(options,'verbose'))
        verbose = options.verbose;
    else
        verbose = false;
    end
    if (isfield(options,'figure_ID'))
        % Field exists
        figure_ID = options.figure_ID;
        if (ishandle(figure_ID))
            % Handle
            while (~strcmp(get(figure_ID,'Type'),'figure'))
                % Not a figure
                figure_ID = figure_ID + 1;
                figure(figure_ID);
            end
        else
            % Handle
            figure(figure_ID);
        end
    else
        % Field does not exist
        figure_ID = figure;
    end

end % process_options
