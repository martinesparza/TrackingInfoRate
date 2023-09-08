import argparse
import re

import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
import glob


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folder", type=str, default='./')
    parser.add_argument("-c", "--cond", type=int, default=1)
    args = parser.parse_args()

    files = glob.glob(f"{args.folder}/*/*.csv")

    df = pd.DataFrame()
    for file in files:
        # Wrap subject number
        file_name = re.split('/', string=file)[-1]
        if len(file_name) == 28:
            participant = int(file_name[10:11])
        else:
            participant = int(file_name[10:12])

        # Wrap condition
        condition = int(file_name[-5:-4])
        if condition < 4:
            reinforcement = 'OFF'
        else:
            reinforcement = 'ON'

        if (condition == 1) or (condition == 4):
            uncertainty = 'Low'
        elif (condition == 2) or (condition == 5):
            uncertainty = 'Medium'
        elif (condition == 3) or (condition == 6):
            uncertainty = 'High'

        tmp_df = pd.read_csv(file)
        tmp_df['Participant'] = participant
        tmp_df['Reinforcement'] = reinforcement
        tmp_df['Uncertainty'] = uncertainty

        df = pd.concat((df, tmp_df))
    # breakpoint()

    # Stats
    md = smf.mixedlm("TargetFeedforward ~ "
                     "TrialNumber*Reinforcement*Uncertainty",
                     df,
                     # re_formula="~TrialNumber",
                     groups=df["Participant"])
    mdf = md.fit(method=["powell"])
    print(mdf.summary())

    return

if __name__ == '__main__':
    main()