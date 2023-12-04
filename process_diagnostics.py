import argparse
import os
import re
from dataclasses import dataclass

import matplotlib.pyplot as plt
import scipy
import seaborn as sns
import pandas as pd
import glob
import numpy as np



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", type=str,
                        default='./output_emg_v2/diagnostics.csv')
    args = parser.parse_args()

    df = pd.read_csv(args.file)

    pre_idx = np.squeeze(np.where((df['RemovedTrial']).isin(
        np.arange(6, 9).tolist())))
    train_idx = np.squeeze(np.where((df['RemovedTrial']).isin(
        np.arange(9, 27).tolist())))
    post_idx = np.squeeze(np.where((df['RemovedTrial']).isin(
        np.arange(27, 36).tolist())))

    df.loc[pre_idx,'Interval'] = 'Pre'
    df.loc[post_idx, 'Interval'] = 'Post'
    df.loc[train_idx, 'Interval'] = 'Train'

    # breakpoint()



    with plt.style.context('default'):
        plt.rcParams["font.family"] = "Arial"
        palette = 'flare'

        sns.set_context('talk')
        fig, ax = plt.subplots(3, 1, figsize=(10, 10))
        sns.countplot(df, x="Subject", ax=ax[0], palette=palette,
                      hue='Interval')

        sns.countplot(df, x="Condition", ax=ax[1], palette=palette,
                      hue='Interval')

        sns.countplot(df, x="Block", ax=ax[2], palette=palette,
                      hue='Interval')

    plt.tight_layout()
    plt.savefig(f"./diagnostics/overall_diagnostics.png")

    subj_2_inspect = np.squeeze(np.where(df['Subject'].isin(np.arange(28,
                                                                      39))))
    # breakpoint()
    df_subset = df.loc[subj_2_inspect]
    with plt.style.context('default'):
        plt.rcParams["font.family"] = "Arial"
        palette = 'flare'

        sns.set_context('notebook')
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.countplot(df_subset, x="Subject", ax=ax, palette=palette,
                      hue='Condition')
        ax.set_ylabel('# removed trials')
    plt.tight_layout()
    plt.savefig(f"./diagnostics/subset.png")



    return

if __name__ == '__main__':
    main()
