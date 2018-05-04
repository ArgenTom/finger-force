% FORM_AAMFINGER_MODEL Calibrates all three AAM-based models given a matrix
% of force information (force_matrix) and matrices of the Shape, Texture, and
% Appearance parameters from the AAM.  Each model assumes the force is a
% linear combination of the respective parameters.
% 
% Written by Thomas R. Grieve
% 4 February 2013
% University of Utah
%
% 

function [shape_model, texture_model, appearance_model] = form_AAMFinger_model(force_matrix, ShapeParams, TextureParams, AppearanceParams)

    % Determine size of inputs
    num_images = size(AppearanceParams,1);
    
    % Use Multiple Linear Regression (i.e., Least Squares) to find the
    % parameter weights that correlate the Shape, Texture and Appearance
    % parameters to the forces
    shape_model.param_weights = [ones(num_images,1) ShapeParams] \ force_matrix;
    texture_model.param_weights = [ones(num_images,1) TextureParams] \ force_matrix;
    appearance_model.param_weights = [ones(num_images,1) AppearanceParams] \ force_matrix;

end % form_AAMFinger_model
