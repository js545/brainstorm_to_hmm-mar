%% Working within HMM-MAR / OSL

data1 = ft_read_data('/home/nebmeg/Documents/HMM_T2DM_Data/006c_ecr_raw_tsss_mc.fif');
data1 = data1';
data2 = ft_read_data('/home/nebmeg/Documents/HMM_T2DM_Data/010c_ecr_raw_tsss_mc.fif');
data2 = data2';

data_concatenate = cat(1, data1, data2);
T = [size(data1, 1), size(data2, 1)];

options = struct();
options.K = 5;

hmmmar(data_concatenate, T, options)


%% Working from Brainstorm timeseries outputs (per parcel)

data = load('/home/nebmeg/Documents/HMM_T2DM_Data/timeseries/006c_DK_scout_timeseries_epochs_removed');
% Invert Brainstorm format to match HMM-MAR
timeseries = data.Value';

% Select data subset just for training / testing
ts_subset = timeseries(1:1000, :);

T = size(ts_subset, 1);

options = struct();
options.K = 10;

[hmm, Gamma, Xi, vpath] = hmmmar(ts_subset, T, options);








