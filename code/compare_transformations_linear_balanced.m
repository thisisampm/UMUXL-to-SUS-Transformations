function compare_transformations_linear_balanced(A, A2, C)
% compare_transformations_linear_balanced
% Finds a linear A3 = a*A + b that improves RMSE over A2 and has better
% pattern match (vs SUS) than A1, using a sweep and mixed-effects analysis.
%
% Inputs:
%   A  - [ParticipantID, Day, Raw Score]
%   A2 - [ParticipantID, Day, Score] (structure-aligned transformation)
%   C  - [ParticipantID, Day, SUS Score]

% === Setup ===
participantID = A(:,1);
day = A(:,2);
rawA = A(:,3);
SUS = C(:,3);
A2_scores = A2(:,3);
A1 = 22.9 + 0.65 * ((rawA - 2) * (100 / 12));

% === Sweep Linear Weights ===
w_range = linspace(0, 1.5, 40);
results = sweep_linear_tradeoff_weights(A, A2, C, w_range);

% === Choose Best A3 (if any passed filter) ===
if isempty(results.bestParams)
    error('No A3 passed filter criteria.');
end
params_A3 = results.bestParams;
A3 = params_A3(1) * rawA + params_A3(2);

% === RMSE ===
rmse_A1 = sqrt(mean((A1 - SUS).^2));
rmse_A2 = sqrt(mean((A2_scores - SUS).^2));
rmse_A3 = sqrt(mean((A3 - SUS).^2));

% === Mixed-Effects Evaluation ===
dataTable = table(participantID, day, SUS, A1, A2_scores, A3, ...
    'VariableNames', {'ParticipantID','Day','SUS','A1','A2','A3'});
dataTable.diffA1 = dataTable.A1 - dataTable.A2;
dataTable.diffA3 = dataTable.A3 - dataTable.A2;

model_A1 = fitlme(dataTable, 'SUS ~ diffA1 + (1|ParticipantID)');
model_A3 = fitlme(dataTable, 'SUS ~ diffA3 + (1|ParticipantID)');

slope_A1 = model_A1.Coefficients.Estimate(2);
pval_A1  = model_A1.Coefficients.pValue(2);
slope_A3 = model_A3.Coefficients.Estimate(2);
pval_A3  = model_A3.Coefficients.pValue(2);

% === Plot: Transformed Scores vs SUS ===
fig1 = figure('Visible','on'); hold on;
scatter(SUS, A1, 36, 'r', 'o');
scatter(SUS, A2_scores, 36, 'g', 'o');
scatter(SUS, A3, 36, 'b', 'o');
plot(SUS, predict(fitlm(SUS, A1)), 'r-', 'LineWidth', 1.5);
plot(SUS, predict(fitlm(SUS, A2_scores)), 'g-', 'LineWidth', 1.5);
plot(SUS, predict(fitlm(SUS, A3)), 'b-', 'LineWidth', 1.5);
plot([0 100], [0 100], 'k--');
xlabel('SUS'); ylabel('Transformed Score');
title('Transformed Scores vs SUS');
xlim([0 100]); ylim([0 100]); axis square; grid on;
legend({'A1','A2','A3','Fit A1','Fit A2','Fit A3','Unity'}, 'Location', 'northwest');
saveas(fig1, fullfile('figures', 'UMUXL_vs_SUS_Comparison.png'));
close(fig1);

% === Plot RMSE Comparison ===
participants = unique(participantID);
n = numel(participants);
rmse_A1_ind = zeros(n,1);
rmse_A2_ind = zeros(n,1);
rmse_A3_ind = zeros(n,1);
for i = 1:n
    idx = participantID == participants(i);
    rmse_A1_ind(i) = sqrt(mean((A1(idx) - SUS(idx)).^2));
    rmse_A2_ind(i) = sqrt(mean((A2_scores(idx) - SUS(idx)).^2));
    rmse_A3_ind(i) = sqrt(mean((A3(idx) - SUS(idx)).^2));
end
data = [rmse_A1_ind, rmse_A2_ind, rmse_A3_ind];
means = mean(data); se = std(data) ./ sqrt(n);
ci80 = tinv(0.9, n-1) * se;
x = [1, 2, 3];
jitter = 0.08;
colors = [1 0.4 0.4; 0.4 0.8 0.4; 0.4 0.4 1];
fig2 = figure('Visible','on'); hold on;
for i = 1:3
    scatter(x(i) + randn(n,1)*jitter, data(:,i), 36, colors(i,:), 'o', 'MarkerEdgeAlpha', 0.5);
end
x_jittered = zeros(n, 3);
for i = 1:3
    x_jittered(:, i) = x(i) + randn(n,1)*jitter;
    scatter(x_jittered(:,i), data(:,i), 36, colors(i,:), 'o', 'MarkerEdgeAlpha', 0.5);
end

% Draw connecting lines between participant-matched points
for i = 1:n
    plot(x_jittered(i,:), data(i,:), '-', 'Color', [0.7 0.7 0.7 0.3]);
end
errorbar(x, means, ci80, 'k', 'LineStyle', 'none', 'LineWidth', 1.5);
scatter(x, means, 50, 'k', 'filled');
set(gca, 'XTick', x, 'XTickLabel', {'Legacy','Structure','Balanced'});
ylabel('Participant-level RMSE');
title('RMSE Comparison: A1, A2, A3');
axis square; grid off;
saveas(fig2, fullfile('figures', 'RMSE_Comparison.png'));
close(fig2);

% === Print Summary ===
fprintf('\n=== Transformation Comparison Summary ===\n');
fprintf('RMSE:\n');
fprintf('  A1 (Legacy)    : %.2f\n', rmse_A1);
fprintf('  A2 (Structure) : %.2f\n', rmse_A2);
fprintf('  A3 (Balanced)  : %.2f\n', rmse_A3);
fprintf('\nMixed-Effects Pattern Alignment:\n');
fprintf('  A1 - A2 → SUS  : Slope = %.2f, p = %.4f\n', slope_A1, pval_A1);
fprintf('  A3 - A2 → SUS  : Slope = %.2f, p = %.4f\n', slope_A3, pval_A3);
fprintf('\nA3 Formula: A3 = %.3f * A + %.3f\n', params_A3(1), params_A3(2));
fprintf('Best tradeoff weight used: %.2f\n', results.w_range(results.bestIdx));
fprintf('========================================\n');
end
