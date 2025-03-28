function results = sweep_linear_tradeoff_weights(A, A2, C, w_range)
% sweep_linear_tradeoff_weights
% Finds a linear A3 = a*A + b that balances RMSE and structure
% Returns sweep results and identifies any A3 that:
%   - Has RMSE < A2
%   - Has better structural alignment than A1 (lower abs(slope))
%
% Saves RMSE vs slope plot with acceptable candidates highlighted

% === Setup ===
rawA = A(:,3);
SUS = C(:,3);
A2_scores = A2(:,3);
participantID = A(:,1);

% A1 definition
A1 = 22.9 + 0.65 * ((rawA - 2) * (100 / 12));
slope_A1 = fitlm(A1 - A2_scores, SUS).Coefficients.Estimate(2);
rmse_A2 = sqrt(mean((A2_scores - SUS).^2));

nW = length(w_range);
params_all = nan(nW, 2);  % [a, b]
RMSEs = nan(nW, 1);
slopes = nan(nW, 1);
costs = nan(nW, 1);
acceptable = false(nW,1);

opts = optimset('Display','off', 'MaxIter', 1000, 'MaxFunEvals', 1000);

for i = 1:nW
    w = w_range(i);

    costFcn = @(p) local_cost(p, rawA, A2_scores, SUS, w);
    bestP = fminsearch(costFcn, [1, 0], opts);

    a = bestP(1); b = bestP(2);
    A3 = a * rawA + b;

    % Validate
    if any(isnan(A3)) || any(isinf(A3)) || std(A3) < 1e-3
        continue;
    end

    diffA = A3 - A2_scores;
    if std(diffA) < 1e-3
        continue;
    end

    lm = fitlm(diffA, SUS);
    slope = lm.Coefficients.Estimate(2);
    rmse = sqrt(mean((A3 - SUS).^2));
    range_penalty = std(A3);
    cost = rmse - w * abs(slope) + 0.1 * range_penalty;

    params_all(i,:) = bestP;
    RMSEs(i) = rmse;
    slopes(i) = slope;
    costs(i) = cost;

    % Evaluate success condition
    if rmse < rmse_A2 && abs(slope) < abs(slope_A1)
        acceptable(i) = true;
    end
end

% Store results
results.w_range = w_range(:);
results.params = params_all;
results.RMSE = RMSEs;
results.slope = slopes;
results.cost = costs;
results.acceptable = acceptable;

% === Plot and Save ===
fig = figure('Visible', 'off'); hold on;
scatter(RMSEs, slopes, 40, w_range, 'filled');
colormap(parula);
colorbar; caxis([min(w_range), max(w_range)]);
xlabel('RMSE'); ylabel('Slope (A3 - A2 → SUS)');
title('Trade-off between Accuracy and Pattern Alignment');
axis square;

% Highlight acceptable points
scatter(RMSEs(acceptable), slopes(acceptable), 60, 'k', 'filled');

% Save plot
outdir = fullfile(pwd, 'figures');
if ~exist(outdir, 'dir'); mkdir(outdir); end
saveas(fig, fullfile(outdir, 'RMSE_vs_Slope_Sweep.png'));
close(fig);

% === Best Acceptable A3 (if any) ===
if any(acceptable)
    [~, bestIdx] = min(RMSEs + abs(slopes)); % Balanced min of both
    results.bestIdx = bestIdx;
    results.bestParams = params_all(bestIdx,:);
    fprintf('\n✅ Best balanced solution found:\n');
    fprintf('Weight: %.2f\n', w_range(bestIdx));
    fprintf('A3 = %.3f * A + %.3f\n', results.bestParams(1), results.bestParams(2));
    fprintf('RMSE: %.2f, Slope: %.2f\n', RMSEs(bestIdx), slopes(bestIdx));
else
    results.bestIdx = NaN;
    results.bestParams = [];
    fprintf('\n⚠️  No linear solution met RMSE < A2 and slope < A1.\n');
end

end

function cost = local_cost(p, A, A2, SUS, w)
a = p(1);
b = p(2);

% Reject unreasonable params
if a < 0.1 || a > 10 || b < -50 || b > 150
    cost = Inf; return;
end

A3 = a * A + b;
if any(isnan(A3)) || any(isinf(A3)) || std(A3) < 1e-3
    cost = Inf; return;
end

diffA = A3 - A2;
if std(diffA) < 1e-3
    cost = Inf; return;
end

lm = fitlm(diffA, SUS);
slope = lm.Coefficients.Estimate(2);
rmse = sqrt(mean((A3 - SUS).^2));
range_penalty = std(A3);

cost = rmse - w * abs(slope) + 0.1 * range_penalty;
end
