%% Load in data and format

clear
clc

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

save('participant_data.mat', 'ts_concat_cell', 'T_cell')

%% Initialize options for TDE hmm
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

%% Run HMM

% When running into "too many input arguments" error, likely duplicate
% paths to downloaded versions of HMM-MAR (may be a second copy through OSL
% package download

for rep = 1:2

    savename = '/home/nebmeg/Data/T2D_RS_Markov/Outputs/k' + string(K) + ...
        '_lag' + string(lag) + '_rep' + string(rep) + '.mat';
    [hmm, Gamma, Xi, vpath] = hmmmar(ts_concat_cell, T_cell, options);

    save(savename,'hmm', 'Gamma', 'vpath')

end 


%% Spectral Estimation using Zhang 2021 Publication GitHub

K = 5;
lag = 5;

pad_options = struct;
pad_options.embeddedlags = -lag:lag;

spec_options = struct();
% spec_options.fpass determines the first index of the vector (psd, coh)
% Can use options.Nf if using hmmspectramar() = parametric equivalent of
% hmmspectramt
% high_freq = Frequency cutoff for bandpass
high_freq = 50;
spec_options.fpass = [1 high_freq];
spec_options.p = 0; % no confidence intervals
spec_options.to_do = [1 0]; % no pdc
spec_options.win = 256;
spec_options.embeddedlags = -5:5;
spec_options.Fs = 250;

num_parc = 68;

% N = number of participants
N = 4; 

load('participant_data.mat')

for rep = 1:3
    
    load('/DATA/T2D_RS_Markov/Outputs/k5_lag5_rep' + string(rep) + '.mat', 'hmm', 'Gamma', 'vpath', 'T_cell')
    
    Gamma = padGamma(Gamma, T_cell, pad_options);

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

    % Create net mean activation maps

    net_mean = zeros(num_parc,size(psd,2));
    for f = size(psd,1)
        for kk = 1:size(psd,2)
            tmp = squeeze(mean(mean(abs(psd(:,kk,1:high_freq,:,:)), 3), 1));
            net_mean(:,kk) = zscore(diag(tmp));
        end
    end


    save('rep' + string(rep) + '_net_mean_activations.mat', 'net_mean')

end





