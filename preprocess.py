"""
Author: @mesparza
Script for preprocessing subject data
"""
import glob
import os
import pickle

import pandas as pd
import scipy
import numpy as np

from __init__ import PATHS

import warnings
warnings.filterwarnings("ignore")


def main():
    # Iterate subject folder
    sub_folders = np.sort(glob.glob(f"{PATHS.data_path}/81Y*"))

    for sub_folder in sub_folders[0:1]:
        # Iterate subject file
        sub_files = np.sort(glob.glob(f"{sub_folder}/rml_FTT_B*.mat"))

        for sub_file in sub_files[3:4]:
            mat = scipy.io.loadmat(sub_file)
            cond = mat['cond'][0][0]
            block = mat['block'][0][0]
            Color_target_alltrials = mat['Color_target_alltrials']
            Seq_target = mat['Seq_target']
            CURSOR = mat['CURSOR']

            if CURSOR.shape[1] > 6:
                breakpoint()
                # if Color_target_alltrials == [0.8302, 0.5, 1]):
                #     colorOK = [0.5, 0.9495, 1]
                #     reinforced = False
                #
                # elif any((Color_target_alltrials == [1, 0.5, 0.5]).all(
                #         axis=1).flatten()):
                #     colorOK = [0.5, 1, 0.5]
                #     reinforced = True

                # else:
                #     colorOK = [0.6, 0.6, 0.6]
                #     reinforced = None

                if reinforced is not None:
                    colorFB = (Color_target_alltrials == colorOK).all(axis=1)

                    # Create dictionary
                    out = {'data': pd.DataFrame({'cursor': CURSOR[0:420, 5:],
                                                 'colour': colorFB[0:420, 5:],
                                                 'target': Seq_target[
                                                           0:420].repeat(30)}),
                           'block': block, 'cond': cond}

                    # Save to pickle file
                    file_name = (f"prepro_81Y{mat['subj']}_block-{block}_con"
                                 f"d-{cond}")
                    with open(f"{PATHS.preprocessed_data_path}"
                              f"/{file_name}.pkl",
                              'wb') as f:
                        pickle.dump(out, f)


if __name__ == '__main__':
    main()
