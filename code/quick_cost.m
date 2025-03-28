function cost = quick_cost(p, A, A2, SUS, w)
% quick_cost
% Computes the cost of a power-law A3 transformation for a given parameter set:
% A3 = a * A^p + b
%
% Inputs:
%   p   - [a, power, b] vector of parameters
%   A   - Raw A scores (Nx1 vector)
%   A2  - Transformed A2 scores for slope comparison
%   SUS - Target SUS scores
%   w   - Weight for pattern alignment in cost function
%
% Output:
%   cost - RMSE - w * abs(slope), or Inf if invalid

a = p(1);
power = p(2);
b = p(3);

% Avoid numerical instability
if a <= 0 || isnan(power) || abs(power) > 10 || isnan(b)
    cost = Inf;
    return;
end

% Compute nonlinear A3
A3 = a * (A .^ power) + b;

% Skip degenerate A3
if any(isnan(A3)) || any(isinf(A3)) || std(A3) < 1e-3
    cost = Inf;
    return;
end

% RMSE with SUS
rmse = sqrt(mean((A3 - SUS).^2));

% Pattern match slope: A3 - A2 vs SUS
diffA = A3 - A2;

% Skip if difference is constant
if std(diffA) < 1e-3
    cost = Inf;
    return;
end

warnState = warning('off', 'all');
lm = fitlm(diffA, SUS);
warning(warnState);
slope = lm.Coefficients.Estimate(2);

% Combined cost
cost = rmse - w * abs(slope);
end
