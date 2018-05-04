function fit_stats = find_linear_fit_stats(data1, data2, options)

    % Determine optional inputs
    if (nargin() == 2)
        [figure_ID, verbose, debug_mode] = process_options([]);
    elseif (nargin() == 3)
        [figure_ID, verbose, debug_mode] = process_options(options);
    else
        error('Must have 2 or 3 inputs!');
    end
    
    % Determine size of inputs
    [num_data, num_vars] = size(data1);
    if (size(data2,1) ~= num_data)
        error('Size of input data is not consistent!');
    end
    if (size(data2,2) < num_vars)
        % Only process up to the minimum number of variables, whether it is
        %   the number of columns in data1 or data2
        num_vars = size(data2,2);
    end
    
    % Form output structure
    fit_stats = struct('slope',[],'intercept',[],'syx',[],'rsq',[],'St',[],'Sr',[],'xbar',[],'ybar',[]);
    
    % Calculate means and standard deviations of input data
    for varIdx = num_vars:-1:1
        % Extract the (x,y) coordinates
        x = data1(:,varIdx);
        y = data2(:,varIdx);
        
        % Calculate the means of both variables
        fit_stats(varIdx).xbar = mean(x);
        fit_stats(varIdx).ybar = mean(y);
        
        % Perform a linear fit
        fit_stats(varIdx).slope = (x'*y - sum(x)*sum(y)/num_data) / (x'*x - sum(x)*sum(x)/num_data);
        fit_stats(varIdx).intercept = fit_stats(varIdx).ybar - fit_stats(varIdx).slope * fit_stats(varIdx).xbar;
        
        % Calculate the total error
        actuals = y - fit_stats(varIdx).ybar;
        fit_stats(varIdx).St = actuals' * actuals;
        
        % Calculate the total error
        estimates = fit_stats(varIdx).slope * x + fit_stats(varIdx).intercept;
        fit_stats(varIdx).Sr = (y - estimates)' * (y - estimates);
        
        % Calculate the standard error of the estimate
        fit_stats(varIdx).syx = sqrt(fit_stats(varIdx).Sr / (num_data-2));
        
        % Calculate the variance
        fit_stats(varIdx).rsq = (fit_stats(varIdx).St - fit_stats(varIdx).Sr) / fit_stats(varIdx).St;
        
        if (debug_mode)
            figure(figure_ID);
            hold on;
            plot(x,y,'k.',x, estimates,'r-');
            hold off;
        end
    end % varIdx
    
    if (debug_mode)
        axis equal;
    end

end % find_linear_fit_stats

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Process the input options and assign default values as needed
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [figure_ID, verbose, debug_mode] = process_options(options)

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
    if (debug_mode)
        if (isfield(options,'figure_ID'))
            figure_ID = options.figure_ID;
            if ((ishandle(figure_ID)) && (strcmp(get(figure_ID,'Type'),'figure')))
                figure(figure_ID);
            else
                figure_ID = figure;
            end
        else
            figure_ID = figure;
        end
    else
        figure_ID = false;
    end

end % process_options
