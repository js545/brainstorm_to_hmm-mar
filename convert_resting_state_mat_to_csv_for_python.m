%% Convert .mat states to csv

filename='E:/GitHub_Master/brainstorm_to_hmm-mar/resting_states_test/rep3_net_mean_activations.mat';
load(filename);

csvwrite('E:/GitHub_Master/brainstorm_to_hmm-mar/resting_states_test/rep3_net_mean_activations.csv', net_mean)

%% Need to add colnames and index for IDs

%% Then convert back to atlas

clearvars

load('C:/Users/sonjak/Desktop/scout_Desikan-Killiany_68.mat');
t = readtable('E:/GitHub_Master/brainstorm_to_hmm-mar/resting_states_test/state_atlases/state5_rep3.csv');
t = table2array(t);

for i = 1:length(Scouts)

    Scouts(i).Color = [t(i, 1), t(i, 2), t(i, 3)];

end

Name = [Name '_rep3_state5'];

save scout_DK68_rep3_state5.mat Name Scouts TessNbVertices -mat






