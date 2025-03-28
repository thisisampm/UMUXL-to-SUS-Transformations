function results = sweep_sigmoid_tradeoff_weights_fast(A, A2, C, w_range)
% sweep_sigmoid_tradeoff_weights_fast
% Quickly evaluates sigmoid-shaped A3 = L / (1 + exp(-k*(A - x0))) + b
% Optimizes for RMSE - w * |slope|  (OLS)
%
% Inputs:
%   A       - [ParticipantID, Day, Raw Score]
%   A2      - [ParticipantID, Day, Score] for comparison
%   C       - [ParticipantID, Day, SUS Score]
%   w_range - vector of weights to sweep (e.g. linspace(0, 5, 10))
%
% Output:
%   results - struct with .params, .RMSE, .slope, .w_range

rawA = A(:,3);
A2_score = A2(:,3);
SUS = C(:,3);

nW = length(w_range);
params_all = nan(nW, 4);  % [L, k, x0, b]
RMSEs = nan(nW, 1);
slopes = nan(nW, 1);

opts = optimset('MaxFunEvals', 1000, 'MaxIter', 1000, 'Display', 'off');

for i = 1:nW
    w = w_range(i);
    costFcn = @(p) quick_sigmoid_cost(p, rawA, A2_score, SUS, w);
    p0 = [100, 1, mean(rawA), 0];  % [L, k, x0, b]

    try
        bestP = fminsearch(costFcn, p0, opts);
    catch
        continue;
    end

    A3 = bestP(1) ./ (1 + exp(-bestP(2) * (rawA - bestP(3)))) + bestP(4);

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

    warnState = warning('off', 'all');
    lm = fitlm(diffA, SUS);
    warning(warnState);
    slopes(i) = lm.Coefficients.Estimate(2);
end

% Plot results
valid = ~isnan(RMSEs) & ~isnan(slopes);
figure;
scatter(RMSEs(valid), slopes(valid), 50, w_range(valid), 'filled');
colormap(parula);
colorbar;
xlabel('RMSE');
ylabel('Slope (OLS)');
title('Sigmoid A3: RMSE vs Pattern Slope');
axis square;
grid on;

% Output
results.w_range = w_range(:);
results.params = params_all;
results.RMSE = RMSEs;
results.slope = slopes;
end
