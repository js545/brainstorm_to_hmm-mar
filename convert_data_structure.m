% Load data / timeseries
df = load('E:/Results/T2D_Resting_State_Results/Scout_TS/006c_DK_scout_timeseries_epochs_removed.mat');
ts = df.Value;

% Number of scouts in atlas
scout_n = 68;
% Number of timepoints per window
time_window = 4000; 