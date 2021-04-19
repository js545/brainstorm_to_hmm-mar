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


%% Working from Brainstorm timeseries outputs

K=10; %Number of states
Fs = 1000; %Sampling frequency

data = load('/home/nebmeg/Documents/HMM_T2DM_Data/timeseries/converted/006c_DK_scout_timeseries_epochs_removed');
% Invert Brainstorm format to match HMM-MAR
timeseries = data.d_full;

% Select data subset just for training / testing
timeseries = timeseries(1:80000, 1:3);

%T = size(timeseries, 1);
T = 4000 * ones(size(timeseries,1)/4000,1);

options = struct();
options.K = K;

[hmm, Gamma, Xi, vpath] = hmmmar(timeseries, T, options);

%% Visualize Output

subplot(2,1,1)
plot(Gamma(1:1000,:)), set(gca,'Title',text('String','True state path'))
set(gca,'ylim',[-0.2 1.2]); ylabel('state #')

subplot(2,1,2)
plot(vpath(1:1000)), set(gca,'Title',text('String','True state path'))
set(gca,'ylim',[0 hmm.K+1]); ylabel('state #')


%% Spectral Info??

options_mt = struct('Fs',Fs);
options_mt.fpass = [1 48];
options_mt.tapers = [4 7];
options_mt.p = 0;
options_mt.win = 500;
options_mt.order = 2;
spectral_info = hmmspectramt(timeseriess,T,Gamma,options_mt);

plot_hmmspectra(spectral_info);

%% State Dynamic Measures

FO = getFractionalOccupancy (Gamma,T,hmm.train); % state fractional occupancies per session
LifeTimes = getStateLifeTimes (Gamma,T,hmm.train); % state life times
Intervals = getStateIntervalTimes (Gamma,T,hmm.train); % interval times between state visits
SwitchingRate =  getSwitchingRate(Gamma,T,hmm.train); % rate of switching between stats

