function results = sweep_saturating_tradeoff_weights_fast(A, A2, C, w_range)
% Fast sweep using saturating A3 = a * A / (A + c) + b
% Uses ordinary least squares for speed (fitlm)
%
% Inputs:
%   A       - [ParticipantID, Day, Raw Score]
%   A2      - [ParticipantID, Day, Score] for "New"
%   C       - [ParticipantID, Day, SUS Score]
%   w_range - vector of weights (e.g. linspace(0, 5, 10))
%
% Output:
%   results - struct with .params, .RMSE, .slope, .w_range

participantID = A(:,1);
rawA = A(:,3);
A2_score = A2(:,3);
SUS = C(:,3);

nW = length(w_range);
params_all = nan(nW, 3);  % [a, c, b]
RMSEs = nan(nW, 1);
slopes = nan(nW, 1);

opts = optimset('MaxFunEvals', 500, 'MaxIter', 500, 'Display', 'off');

for i = 1:nW
    w = w_range(i);
    costFcn = @(p) quick_saturating_cost(p, rawA, A2_score, SUS, w);
    p0 = [100, 5, 0];  % initial guess: [a, c, b]

    try
        bestP = fminsearch(costFcn, p0, opts);
    catch
        continue;
    end

    A3 = bestP(1) * rawA ./ (rawA + bestP(2)) + bestP(3);

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
title('Saturating A3: RMSE vs Pattern Slope');
axis square;
grid on;

% Return results
results.w_range = w_range(:);
results.params = params_all;
results.RMSE = RMSEs;
results.slope = slopes;
end
