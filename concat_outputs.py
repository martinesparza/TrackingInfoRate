import argparse
import glob
import re

import numpy as np
from tqdm import tqdm
import pandas as pd
from plotting import Config81Y, Config83Y, ConfigEMG

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--name", type=str,
                    default='all_subjects.csv')
parser.add_argument("-f", "--folder", type=str, default='data/')
args = parser.parse_args()


def main(args):
    if args.folder[-3:] == '81Y':
        config = Config81Y
    elif args.folder[-3:] == '83Y':
        config = Config83Y
    else:
        config = ConfigEMG

    files = glob.glob(f"{args.folder}/o*.csv")
    df = pd.DataFrame()
    for csv in files:
        file_name = re.split('/', string=csv)[-1]
        df_ = pd.read_csv(csv)
        # breakpoint()
        if len(file_name) == config.name_length[0]:  #
            participant = int(
                file_name[config.name_length[1]:config.name_length[2]])
        else:
            participant = int(
                file_name[config.name_length[1]:config.name_length[3]])
        # Wrap block
        block = int(file_name[-11:-10])

        # Wrap condition
        condition = int(file_name[-5:-4])
        if config.__name__ != 'ConfigEMG':
            if condition < 4:
                reinforcement = 'OFF'
            else:
                reinforcement = 'ON'

            if (condition == 1) or (condition == 4):
                condition_ = config.condition[1][0]  # 20 Hz
            elif (condition == 2) or (condition == 5):
                condition_ = config.condition[1][1]  # 80 Hz
            elif (condition == 3) or (condition == 6):
                condition_ = config.condition[1][2]  # Sham
        else:
            if (condition % 2) == 0:
                reinforcement = 'ON'
            else:
                reinforcement = 'OFF'

            if (condition == 1) or (condition == 2):
                condition_ = config.condition[1][0]
            elif (condition == 3) or (condition == 4):
                condition_ = config.condition[1][1]
            elif (condition == 5) or (condition == 6):
                condition_ = config.condition[1][2]
            elif (condition == 7) or (condition == 8):
                condition_ = config.condition[1][3]


        df_['Participant'] = participant
        df_['Reinforcement'] = reinforcement
        df_[f"{config.condition[0]}"] = condition_
        df_['Block'] = block
        df_['Condition'] = condition


        df_['NormTargetFeedback'] = (df_['TargetFeedback'] /
                                     df_['TargetFeedback'].iloc[
                                     config.norm_trials].mean() *
                                     100)
        df_['NormTargetFeedforward'] = (df_['TargetFeedforward'] /
                                        df_['TargetFeedforward'].iloc[
                                        config.norm_trials].mean() * 100)
        df_['NormTargetTotalInfo'] = (df_['TargetTotalInfo'] /
                                      df_['TargetTotalInfo'].iloc[
                                      config.norm_trials].mean() * 100)

        df = pd.concat((df, df_))

    df.to_csv(f"{args.folder}/{args.name}")


if __name__ == '__main__':
    main(args)
