function results = sweep_nonlinear_tradeoff_weights_fast(A, A2, C, w_range)
% sweep_nonlinear_tradeoff_weights_fast
% 
% Fast preview of nonlinear A3 optimization using a basic linear regression.
% Transformation: A3 = a * A^p + b
% Cost = RMSE - w * |slope|  (slope from SUS ~ diffA)
% 
% This version avoids mixed-effects models and is intended for testing ideas quickly.
% 
% Inputs:
%   A       - [ParticipantID, Day, Raw Score]
%   A2      - [ParticipantID, Day, Score] for "New"
%   C       - [ParticipantID, Day, SUS Score]
%   w_range - vector of weights to sweep (e.g. linspace(0, 5, 10))
%
% Output:
%   results - struct with .params, .RMSE, .slope, .w_range

participantID = A(:,1);
rawA = A(:,3);
A2_score = A2(:,3);
SUS = C(:,3);

nW = length(w_range);
params_all = nan(nW, 3);  % [a, p, b]
RMSEs = nan(nW, 1);
slopes = nan(nW, 1);

opts = optimset('MaxFunEvals', 500, 'MaxIter', 500, 'Display', 'off');

for i = 1:nW
    w = w_range(i);
    costFcn = @(p) quick_cost(p, rawA, A2_score, SUS, w);
    p0 = [1, 1, 0];  % initial guess

    try
        bestP = fminsearch(costFcn, p0, opts);
    catch
        continue;
    end

    A3 = bestP(1) * (rawA .^ bestP(2)) + bestP(3);

    if std(A3) < 1e-3 || any(isnan(A3)) || any(isinf(A3))
        continue;
    end

    params_all(i,:) = bestP;
    RMSEs(i) = sqrt(mean((A3 - SUS).^2));

    diffA = A3 - A2_score;
    if std(diffA) < 1e-3 || all(diffA == diffA(1))
        slopes(i) = NaN;
        continue;
    end

    warnState = warning('off', 'all'); % silence rank deficiency warning
    lm = fitlm(diffA, SUS);
    warning(warnState); % restore

    slopes(i) = lm.Coefficients.Estimate(2);  % slope of diffA
end

% Plot slope vs RMSE
valid = ~isnan(RMSEs) & ~isnan(slopes);
figure;
scatter(RMSEs(valid), slopes(valid), 50, w_range(valid), 'filled');
colormap(parula);
colorbar;
xlabel('RMSE');
ylabel('Slope (OLS)');
title('Fast Nonlinear Trade-off: RMSE vs Pattern Slope');
axis square;
grid on;

% Return results
results.w_range = w_range(:);
results.params = params_all;
results.RMSE = RMSEs;
results.slope = slopes;
end

function cost = quick_cost(p, A, A2, SUS, w)
% Compute cost = RMSE - w * |slope| for nonlinear A3 = a*A^p + b
a = p(1); pow = p(2); b = p(3);

if pow <= 0 || pow > 5 || any(A < 0 & mod(pow, 1) ~= 0)
    cost = Inf;
    return;
end

A3 = a * (A .^ pow) + b;

if std(A3) < 1e-3 || any(isnan(A3)) || any(isinf(A3))
    cost = Inf;
    return;
end

rmse = sqrt(mean((A3 - SUS).^2));

diffA = A3 - A2;
if std(diffA) < 1e-3 || all(diffA == diffA(1))
    cost = Inf;
    return;
end

warnState = warning('off', 'all'); % suppress rank warnings
lm = fitlm(diffA, SUS);
warning(warnState);  % restore

slope = lm.Coefficients.Estimate(2);
cost = rmse - w * abs(slope);
end
