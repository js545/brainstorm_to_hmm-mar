% Load data / timeseries
df = load('E:/Results/T2D_Resting_State_Results/Scout_TS/006c_DK_scout_timeseries_epochs_removed.mat');
ts = df.Value;

% Number of scouts in atlas
n_scout = 68;
% Number of timepoints per window
n_timepoints = 4000; 
% Calculate number of epochs in sample
n_epochs = size(ts,1) / n_scout;

d_full = [];

for i = 1:n_epochs
    
    d_temp = ts(((i-1)*n_scout + 1):(i*n_scout), 1:n_timepoints)';
    d_full = [d_full; d_temp];
    
end