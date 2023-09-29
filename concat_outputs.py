import argparse
import glob
import re

import numpy as np
from tqdm import tqdm
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--name", type=str, default='concat.csv')
parser.add_argument("-p", "--path", type=str, default='data/')
args = parser.parse_args()

def main(args):
    files = glob.glob(f"{args.path}/output*.csv")
    df = pd.DataFrame()
    for file in tqdm(files):
        df_ = pd.read_csv(file)

        file_name = re.split('/', string=file)[-1]
        if len(file_name) == 28:
            participant = int(file_name[10:11])
        else:
            participant = int(file_name[10:12])
        # Wrap block
        block = int(file_name[-11:-10])

        # Wrap condition
        condition = int(file_name[-5:-4])
        if condition < 4:
            reinforcement = 'OFF'
        else:
            reinforcement = 'ON'

        if (condition == 1) or (condition == 4):
            stim = '20 Hz'
        elif (condition == 2) or (condition == 5):
            stim = '80 Hz'
        elif (condition == 3) or (condition == 6):
            stim = 'Sham'

        df_['Participant'] = participant
        df_['Reinforcement'] = reinforcement
        df_['Stimulation'] = stim
        df_['Block'] = block
        df_['NormTargetFeedback'] = (df_['TargetFeedback'] /
                                     df_['TargetFeedback'].iloc[:4].mean() *
                                     100)
        df_['NormTargetFeedforward'] = (df_['TargetFeedforward'] /
                                        df_['TargetFeedforward'].iloc[
                                        :4].mean() * 100)

        df = pd.concat((df, df_))

    df.to_csv(f"{args.path}/{args.name}")



if __name__ == '__main__':
    main(args)
