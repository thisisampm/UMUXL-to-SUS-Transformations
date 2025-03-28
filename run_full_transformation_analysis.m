
% run_full_transformation_analysis.m
% Walks through all transformation trials from A to SUS:
% - Defines A1 and A2
% - Explores linear and nonlinear A3 attempts
% - Visualizes trial-and-error steps
% - Saves figures and summary to GitHub-style structure

% Step 0: Load or define your data matrices
% A, A2, and C must be loaded into the workspace beforehand
% A  = [ParticipantID, Day, RawA]
% A2 = [ParticipantID, Day, Score]  % Structure-aligned transform
% C  = [ParticipantID, Day, SUS]

% Placeholder: replace with your own data loading if needed
load('starting_data.mat');  % if stored externally

% Add /code folder to path
addpath(genpath(fullfile(pwd, 'code')));

%% Step 1: Define A1 and A2
rawA = A(:,3);
SUS = C(:,3);
A1 = 22.9 + 0.65 * ((rawA - 2) * (100 / 12));
A2_scores = A2(:,3);

% Save A vs SUS
figure;
hold on; plot([0 100], [2 14], 'k--')
scatter(SUS,rawA, 36, 'o', 'MarkerEdgeColor', [0.6 0.6 0.6]);
ylabel('Raw UMUXL'); xlabel('SUS');
title('Raw UMUXL vs SUS');
axis square; axis([0 100 2 14])
grid off;
saveas(gcf, fullfile('figures', 'UMUXL_vs_SUS.png'));


%% Step 2: Early A3 Exploration (Trial & Error)

% All of these generate plots and can help visualize dead ends
% Each sweep result is optional, but their figures should be saved to /figures/

% Linear sweep only for RMSE
results_linear = sweep_linear_tradeoff_weights(A, A2, C, linspace(0, 5, 20));

% Nonlinear (power) A3 sweep
results_nonlin = sweep_nonlinear_tradeoff_weights_fast(A, A2, C, linspace(0, 5, 20));

% Sigmoid sweep
results_sigmoid = sweep_sigmoid_tradeoff_weights_fast(A, A2, C, linspace(0, 5, 20));

% Saturating sweep
results_sat = sweep_saturating_tradeoff_weights_fast(A, A2, C, linspace(0, 5, 20));


%% Step 3: Final best linear A3; Save summary to results/
diary(fullfile('results', 'summary.txt'));
compare_transformations_linear_balanced(A, A2, C);
diary off;

disp('Analysis complete. Check /figures and /results for outputs.');
