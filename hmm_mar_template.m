%% Working from Brainstorm timeseries outputs

% addpath(genpath('/home/nebmeg/Documents/osl/HMM-MAR'))
addpath(genpath('/home/nebmeg/matlab_external_packages/HMM-MAR'))

% Loading in files
% myFolder = '/home/nebmeg/Documents/HMM_T2DM_Data/timeseries/converted/';
myFolder = '/home/nebmeg/Data/T2D_RS_Markov/Data/All/';
filePattern = fullfile(myFolder, '*.mat');
theFiles = dir(filePattern);

T_bad_epoch_index = load('/home/nebmeg/Data/T2D_RS_Markov/Data/bad_epochs_index_all.mat').T_bad_epoch_index;

ts_concat = [];
ts_concat_cell = {};
T_cell = {};

for k=1:5%length(theFiles)
   
    baseFileName = theFiles(k).name;
    fullFileName = fullfile(theFiles(k).folder, baseFileName);
    fprintf(1, 'Now reading %s\n', fullFileName);
    
    df = load(fullFileName);
    ts = df.d_full;
    
    % Downsample
    ts = ts(1:4:end, :);
    
    ts_concat = cat(1, ts_concat, ts);
    ts_concat_cell{k, 1} = ts;
    
    T_temp = [];
    
    if size(T_bad_epoch_index{k},1) ~= 0
    
        for i = 1:length(T_bad_epoch_index{k})

            if i == 1

                T_temp(i) = (T_bad_epoch_index{k}(i) - 1) * 1000;

            else

                T_temp(i) = ((T_bad_epoch_index{k}(i) - T_bad_epoch_index{k}(i-1))-1) * 1000;

            end

        end
        
    else
        
        T_Temp(i) = length(ts);
        
    end
    
    if sum(T_temp) < length(ts)
        
        T_temp(i+1) = length(ts) - sum(T_temp);
        
    end
    
    T_temp = nonzeros(T_temp);
    T_cell{k,1} = reshape(T_temp, 1, length(T_temp));
    
end

% % Downsample
% ts_concat = ts_concat(1:4:end, :);
% Fs = 250; %Sampling frequency


% % Select data subset just for training / testing
% ts_concat = ts_concat(1:80000, 1:3);

% n_samples_per_epoch = 1000; % 1000 if downsampled to 250Hz, 4000 if not
% T = n_samples_per_epoch * ones(size(ts_concat,1)/n_samples_per_epoch,1);

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
% options.K = K;
options.Fs = 250;
options.order = 0;
options.verbose = 1;
options.zeromean = 1;
options.covtype='full';

% lag=2;
% options.embeddedlags = -lag:lag;
options.pca = size(ts_concat, 2)*2;
% options.pca = size(ts_concat, 2);
options.standardise = 1;
options.standardise_pc = options.standardise;

% stochastic options
options.BIGNinitbatch = 5;
options.BIGNbatch = 5;
options.BIGtol = 1e-7;
options.BIGcyc = 500;
options.BIGundertol_tostop = 5;
options.BIGdelay = 5;
options.BIGforgetrate = 0.7;
options.BIGbase_weights = 0.9;

options.leakagecorr = -1;
options.useParallel = 1;

r=1;
K=5;
lag=5; 

fprintf(1, 'Now analyzing K=%d states with lag of %d\n', K, lag);

options.K = K;
options.embeddedlags = -lag:lag;
savename = '/home/nebmeg/Data/T2D_RS_Markov/Outputs/k' + string(K) + '_lag' + string(lag) + '_rep' + string(r) + '.mat';
[hmm, Gamma, Xi, vpath] = hmmmar(ts_concat_cell, T_cell, options);

save(savename,'hmm', 'Gamma', 'vpath')

%% Plot brain maps

load('/DATA/T2D_RS_Markov/Outputs/k6_lag5_rep1.mat')

maskfile = '/DATA/T2D_RS_Markov/Data/';
parcfile = 5;

maps = makeMap(hmm);

%% Reliability Estimation with Riemann Distances
% % Needs to be in "positive definite matrix" form
% % Is this condition met with sign flipping function?
% % Still returning infinite values after trying with sign flipping function
% 
% num_states=6;
% 
% full_riemann_mat = zeros(num_states, num_states);
% 
% for j = 1:num_states
%     for k = 1:num_states
%         
%         A = getFuncConn(hmm, j);
%         B = getFuncConn(hmm, k);
%         
%         full_riemann_mat(j, k) = sqrt(sum(log(eig(A,B)).^2));
%         
%     end
% end
% 
% K=6;
% iperm=1;
% D = zeros(iperm, K, K); 
% 
% Gamma = vpath_to_stc(vpath,K);
% setxx;
% hmm = obsupdate(T,Gamma,hmm,data.X,XX,XXGXX);
% 
% for k1 = 1:K-1
%     [~,C1] = getFuncConn(hmm,k1);
%     if ~isempty(A), C1 = A' * C1 * A; end 
%     for k2 = k1+1:K
%         [~,C2] = getFuncConn(hmm,k2);
%         if ~isempty(A), C2 = A' * C2 * A; end 
%         d = riemannian_dist(C1,C2);
%         D(iperm,k1,k2) = d; 
%         D(iperm,k2,k1) = d; 
%     end
% end
% 
% hmmtestretest
% options.K=3;
% options.testretest_rep = 500;


%% State Dynamic Measures

FO = getFractionalOccupancy (Gamma,T,hmm.train); % state fractional occupancies per session
LifeTimes = getStateLifeTimes (Gamma,T,hmm.train); % state life times
Intervals = getStateIntervalTimes (Gamma,T,hmm.train); % interval times between state visits
SwitchingRate =  getSwitchingRate(Gamma,T,hmm.train); % rate of switching between stats

outputfile = '/home/nebmeg/Data//Outputs/k4_lag2_pcadouble.mat';

save% stochastic options
options.BIGNinitbatch = 5;
options.BIGNbatch = 5;
options.BIGtol = 1e-7;
options.BIGcyc = 500;
options.BIGundertol_tostop = 5;
options.BIGdelay = 5;
options.BIGforgetrate = 0.7;
options.BIGbase_weights = 0.9;

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
options_mt.order = 0;
options_mt.embeddedlags = -3:3;

spectral_info = hmmspectramt(ts_concat_cell,T_cell,Gamma,options_mt);

plot_hmmspectra(spectral_info);





%% Updated Template Test

addpath(genpath('/home/nebmeg/matlab_external_packages/HMM-MAR'))

% Loading in files
myFolder = '/home/nebmeg/Downloads/temp_hmm_four/';
filePattern = fullfile(myFolder, '*.mat');
theFiles = dir(filePattern);

ts_concat = [];
ts_concat_cell = {};
T_cell = {};

for k=1:length(theFiles)
   
    baseFileName = theFiles(k).name;
    fullFileName = fullfile(theFiles(k).folder, baseFileName);
    fprintf(1, 'Now reading %s\n', fullFileName);
    
    df = load(fullFileName);
    ts = df.X;
    
    ts_concat = cat(1, ts_concat, ts);
    ts_concat_cell{k, 1} = ts;
    
    T_cell{k, 1} = length(ts);
    
end

%% Test TDE Options

% Initialize options for TDE hmm
options = struct;
options.Fs = 250;
options.order = 0;
options.verbose = 1;
options.zeromean = 1;
options.covtype='full';

options.pca = size(ts_concat, 2)*2;
% options.pca = size(ts_concat, 2);
options.standardise = 1;
options.standardise_pc = options.standardise;

% stochastic options
options.BIGNinitbatch = 5;
options.BIGNbatch = 5;
options.BIGtol = 1e-7;
options.BIGcyc = 500;
options.BIGundertol_tostop = 5;
options.BIGdelay = 5;
options.BIGforgetrate = 0.7;
options.BIGbase_weights = 0.9;

options.leakagecorr = -1;
options.useParallel = 1;

K = 5;
lag = 5;

options.K = K;
options.embeddedlags = -lag:lag;

% When running into "too many input arguments" error, likely duplicate
% paths to downloaded versions of HMM-MAR (may be a second copy through OSL
% package download

for rep = 1:2

    savename = '/home/nebmeg/Data/T2D_RS_Markov/Outputs/k' + string(K) + ...
        '_lag' + string(lag) + '_rep' + string(rep) + '.mat';
    [hmm, Gamma, Xi, vpath] = hmmmar(ts_concat_cell, T_cell, options);

    save(savename,'hmm', 'Gamma', 'vpath', 'T_cell')

end 



%% Spectral Factorization

% Wideband
mapfile = [mapsdir '/state_maps_wideband'];
maps = makeSpectralMap(fitmt_group_fact_wb,1,parcfile,maskfile,1,0,mapfile,workbenchdir);
% Per frequency band
for fr = 1:3
    mapfile = [mapsdir '/state_maps_band' num2str(fr)];
    maps = makeSpectralMap(fitmt_group_fact_4b,fr,parcfile,maskfile,1,0,mapfile,workbenchdir);
end

%% Spectral Estimation using Zhang 2021 Publication GitHub

load('/DATA/T2D_RS_Markov/Outputs/k5_lag5_rep1.mat', 'hmm', 'Gamma', 'vpath', 'T_cell')

K = 5;
lag = 5;

pad_options = struct;
pad_options.embeddedlags = -lag:lag;
Gamma = padGamma(Gamma, T_cell, pad_options);

spec_options = struct();
% spec_options.fpass determines the first index of the vector (psd, coh)
% Can use options.Nf if using hmmspectramar() = parametric equivalent of
% hmmspectramt
high_freq = 50;
spec_options.fpass = [1 high_freq];
spec_options.p = 0; % no confidence intervals
spec_options.to_do = [1 0]; % no pdc
spec_options.win = 256;
spec_options.embeddedlags = -5:5;
spec_options.Fs = 250;

num_parc = 68;

%df_test = ts_concat_cell(1, :);
%fit = hmmspectramt(df_test{1, 1}, 89000, Gamma(1:89000, :), spec_options);

% N = number of participants
N = 4; 

% Input from spec_options.fpass
psd = zeros(N, K, high_freq, num_parc, num_parc);
coh = zeros(N, K, high_freq, num_parc, num_parc);

for subj = 1:N
    
    fprintf('Processing subject %d\n', subj)
    
    if subj == 1
    
    subj_low_index = 1;
    subj_high_index = size(ts_concat_cell{subj, 1}, 1);
    
    else
        
        subj_low_index = subj_high_index + 1;
        subj_high_index = subj_high_index + size(ts_concat_cell{subj, 1}, 1);
    
    end 
    
    subj_dim = subj_high_index - subj_low_index + 1;
    
    fit = hmmspectramt(ts_concat_cell{subj, 1}, subj_dim, ...
        Gamma(subj_low_index:subj_high_index, :), spec_options);

    for n = 1:K
        
        fprintf('Processing state %d\n', n)

        psd(subj, n, :, :, :) = fit.state(n).psd;
        coh(subj, n, :, :, :) = fit.state(n).coh;

    end

end

% Requires osl path from osl-core (not osl-master)
% https://github.com/OHBA-analysis/osl-core
parc = parcellation('/home/nebmeg/Downloads/MNI152_T1_2mm_brain.nii.gz');

% Index changes based on size of original psd/coh matrix. In this example
% case, we do not use multiple participants, so add 1 to the index from the
% reference

net_mean = zeros(num_parc,size(psd,2));
for f = size(psd,1)
    for kk = 1:size(psd,2)
        tmp = squeeze(mean(mean(abs(psd(:,kk,1:high_freq,:,:)), 3), 1));
        net_mean(:,kk) = zscore(diag(tmp));
    end
end

parc.osleyes(net_mean)




%%

% Updated pipline 2/28/22

addpath(genpath('/home/nebmeg/matlab_external_packages/HMM-MAR'))

% Loading in files
myFolder = '/home/nebmeg/Downloads/temp_hmm_four/';
filePattern = fullfile(myFolder, '*.mat');
theFiles = dir(filePattern);

ts_concat = [];
ts_concat_cell = {};
T_cell = {};

for k=1:length(theFiles)
   
    baseFileName = theFiles(k).name;
    fullFileName = fullfile(theFiles(k).folder, baseFileName);
    fprintf(1, 'Now reading %s\n', fullFileName);
    
    df = load(fullFileName);
    ts = df.X;
    
    ts_concat = cat(1, ts_concat, ts);
    ts_concat_cell{k, 1} = ts;
    
    T_cell{k, 1} = length(ts);
    
end

% Initialize options for TDE hmm
options = struct;
options.Fs = 250;
options.order = 0;
options.verbose = 1;
options.zeromean = 1;
options.covtype='full';

options.pca = size(ts_concat, 2)*2;
% options.pca = size(ts_concat, 2);
options.standardise = 1;
options.standardise_pc = options.standardise;

% stochastic options
options.BIGNinitbatch = 5;
options.BIGNbatch = 5;
options.BIGtol = 1e-7;
options.BIGcyc = 500;
options.BIGundertol_tostop = 5;
options.BIGdelay = 5;
options.BIGforgetrate = 0.7;
options.BIGbase_weights = 0.9;

options.leakagecorr = -1;
options.useParallel = 1;

K = 5;
lag = 5;

options.K = K;
options.embeddedlags = -lag:lag;

% When running into "too many input arguments" error, likely duplicate
% paths to downloaded versions of HMM-MAR (may be a second copy through OSL
% package download

for rep = 1:2

    savename = '/home/nebmeg/Data/T2D_RS_Markov/Outputs/k' + string(K) + ...
        '_lag' + string(lag) + '_rep' + string(rep) + '.mat';
    [hmm, Gamma, Xi, vpath] = hmmmar(ts_concat_cell, T_cell, options);

    save(savename,'hmm', 'Gamma', 'vpath', 'T_cell')

end 





















