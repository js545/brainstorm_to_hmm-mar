%% Working from Brainstorm timeseries outputs

addpath(genpath('/home/nebmeg/Documents/osl/HMM-MAR'))

% Loading in files
myFolder = '/home/nebmeg/Documents/HMM_T2DM_Data/timeseries/converted/';
filePattern = fullfile(myFolder, '*.mat');
theFiles = dir(filePattern);

ts_concat = [];

for k=1:length(theFiles)
   
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

% %% Test MAR Options
% K=10; %Number of states
% options = struct();
% options.K = K;
% options.order = 0;
% 
% [hmm, Gamma, Xi, vpath] = hmmmar(ts_concat, T, options);
% 
% % Explained variance?
% explained_var = explainedvar_PCA(ts_concat, T_all, options);

%% Test TDE Options

% Initialize options for TDE hmm
% K=4; %Number of states
% options = struct();
options.K = K;
options.Fs = 250;
options.order = 0;
options.verbose = 1;
options.zeromean = 1;

% lag=2;
% options.embeddedlags = -lag:lag;
% options.pca = size(ts_concat, 2)*2;
options.pca = size(ts_concat, 2);
options.standardise = 1;
options.standardise_pc = options.standardise;

for K= 3:6
    for lag = 2:5
        
        options.K = K;
        options.embeddedlags = -lag:lag;
        savename = '/home/nebmeg/Documents/brainstorm_to_hmm-mar/outputs/k' + string(K) + '_lag' + string(lag) + '_pcasingle.mat';
        [hmm, Gamma, Xi, vpath] = hmmmar(ts_concat, T, options);
        
        FO = getFractionalOccupancy (Gamma,T,hmm.train); % state fractional occupancies per session
        LifeTimes = getStateLifeTimes (Gamma,T,hmm.train); % state life times
        Intervals = getStateIntervalTimes (Gamma,T,hmm.train); % interval times between state visits
        SwitchingRate =  getSwitchingRate(Gamma,T,hmm.train); % rate of switching between stats

        save(savename,'LifeTimes','Intervals','FO','SwitchingRate')
        
    end
end

[hmm, Gamma, Xi, vpath] = hmmmar(ts_concat, T, options);

%% State Dynamic Measures

FO = getFractionalOccupancy (Gamma,T,hmm.train); % state fractional occupancies per session
LifeTimes = getStateLifeTimes (Gamma,T,hmm.train); % state life times
Intervals = getStateIntervalTimes (Gamma,T,hmm.train); % interval times between state visits
SwitchingRate =  getSwitchingRate(Gamma,T,hmm.train); % rate of switching between stats

outputfile = '/home/nebmeg/Documents/brainstorm_to_hmm-mar/outputs/k4_lag2_pcadouble.mat';

save(outputfile,'LifeTimes','Intervals','FO','SwitchingRate')

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

spectral_info = hmmspectramt(ts_concat,T,Gamma,options_mt);

plot_hmmspectra(spectral_info);

