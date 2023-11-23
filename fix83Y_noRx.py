import scipy
import numpy as np
import pandas as pd

df = pd.read_csv('./output_83Y/all_subjects_83Y_v3.csv')
mat_file = scipy.io.loadmat('/home/esparza/matfile.mat')
mat_file = mat_file['matfile']
NoRx_col_idx = 12

NoRx_indices = np.where(mat_file[:, NoRx_col_idx] == 1)

# for index, row in df.iterrows():
for idx in NoRx_indices[0]:
    subject = mat_file[idx, 0]
    trial = mat_file[idx, 1]
    block = mat_file[idx, 2]

    index = df.index[(df['Participant'] == subject) &
                     (df['TrialNumber'] == trial) &
                     (df['Block'] == block)]

    df.at[index[0], 'TargetFeedback'] = np.nan
    df.at[index[0], 'TargetFeedforward'] = np.nan
    df.at[index[0], 'TargetTotalInfo'] = np.nan

df.drop(labels=['Unnamed: 0'], inplace=True)
df.to_csv('./output_83Y/all_subjects_83Y_v4.csv', index=False)