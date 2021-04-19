%AUTHORS:            Jake Son
%VERSION HISTORY:    04/19/2021  v1: First working version

%% Initialize workspace
% Specify the folder where the files live.
myFolder = 'E:/Results/T2D_Resting_State_Results/Scout_TS/';

% Get a list of all files in the folder with the desired file name pattern.
filePattern = fullfile(myFolder, '*.mat');
theFiles = dir(filePattern);

%% Initialize variables specific to the atlas / data
% Number of scouts in atlas
n_scout = 68;
% Number of timepoints per window
n_timepoints = 4000; 

%% Loop through files to convert and save
for k = 1 : length(theFiles)
    
    baseFileName = theFiles(k).name;
    fullFileName = fullfile(theFiles(k).folder, baseFileName);
    fprintf(1, 'Now reading %s\n', fullFileName);
    
    df = load(fullFileName);
    ts = df.Value;
    
    % Calculate number of epochs in sample
    n_epochs = size(ts,1) / n_scout;
    
    d_full = [];
    
    for i = 1:n_epochs
       
        d_temp = ts(((i-1)*n_scout + 1):(i*n_scout), 1:n_timepoints)';
        d_full = [d_full; d_temp];
        
    end
    
    outFileName = fullfile('E:/Results/T2D_Resting_State_Results/Scout_TS/converted_TS/', baseFileName);
    save(outFileName, 'd_full')
    
end