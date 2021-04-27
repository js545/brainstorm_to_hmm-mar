%% Working from Brainstorm timeseries outputs

% Loading in files
myFolder = '/home/nebmeg/Documents/HMM_T2DM_Data/timeseries/converted/';
filePattern = fullfile(myFolder, '*.mat');
theFiles = dir(filePattern);

ts_concat = [];

for k=1:5 %length(theFiles)
   
    baseFileName = theFiles(k).name;
    fullFileName = fullfile(theFiles(k).folder, baseFileName);
    fprintf(1, 'Now reading %s\n', fullFileName);
    
    df = load(fullFileName);
    ts = df.d_full;
    
    ts_concat = cat(1, ts_concat, ts);
    
end

% Downsample
ts_concat = ts_concat(1:4:end, :);
Fs = 250; %Sampling frequency


% % Select data subset just for training / testing
% ts_concat = ts_concat(1:80000, 1:3);

n_samples_per_epoch = 1000; % 1000 if downsampled to 250Hz, 4000 if not
T = n_samples_per_epoch * ones(size(ts_concat,1)/n_samples_per_epoch,1);

% Initialize options for hmm
K=10; %Number of states
options = struct();
options.K = K;
options.order = 0;

[hmm, Gamma, Xi, vpath] = hmmmar(ts_concat, T, options);

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

