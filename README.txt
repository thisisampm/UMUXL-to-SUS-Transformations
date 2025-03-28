# A-to-SUS Transformation Analysis

This project explores different mathematical transformations of an input attitude score A to predict the System Usability Scale (SUS). It includes:

- A1: A legacy linear transformation used for interpretation (Lewis et al 2013)
- A2: A structure-aligned transformation designed to match SUS trends (Berkman et al 2016)
- A3: A series of trial transformations (linear and nonlinear) aimed at improving accuracy and alignment (computed here)

## Features
- RMSE and pattern match (slope) evaluation
- Linear, power, sigmoid, and saturating sweep optimization
- Reproducible transformation plots

## Usage

1. Place your input matrices `A`, `A2`, and `C` in the workspace.
   - Each should be formatted as: `[ParticipantID, Day, Score]`
2. Add all `code/` functions to your MATLAB path.
3. Run the full workflow:

run_full_transformation_analysis.m


## Output

- `/figures/` contains all diagnostic and comparison plots
- `/results/summary.txt` contains RMSE, slope, and final interpretation

## Requirements
- MATLAB R2021a or later
- Statistics and Machine Learning Toolbox


