import numpy as np
import pandas as pd
from matplotlib import cm

def pseudocolor(val, minval, maxval):

    val = (val-minval)/(maxval-minval)

    return cm.jet(val)

# df = pd.read_csv('C:/Users/sonjak/Desktop/net_mean_activations_hmm_k5.csv', index_col=0)

df = pd.read_csv('E:/GitHub_Master/brainstorm_to_hmm-mar/resting_states_test/rep3_net_mean_activations.csv', index_col = 'ID')

for j in range(df.shape[1]):

    converted_df = pd.DataFrame(index=range(68), columns=['R', 'G', 'B'])

    state_df = df[df.columns[j]]
    savename = 'E:/GitHub_Master/brainstorm_to_hmm-mar/resting_states_test/state_atlases/state' + str(j+1) + '_rep3.csv'

    for k in range(df.shape[0]):

        converted_df.at[k, 'R'] = pseudocolor(state_df.to_list()[k], -3, 3)[0]
        converted_df.at[k, 'G'] = pseudocolor(state_df.to_list()[k], -3, 3)[1]
        converted_df.at[k, 'B'] = pseudocolor(state_df.to_list()[k], -3, 3)[2]

    converted_df.to_csv(savename, index=False)





