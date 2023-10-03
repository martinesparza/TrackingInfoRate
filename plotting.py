import argparse
import os
import re
from dataclasses import dataclass

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import glob
import numpy as np

import warnings

warnings.filterwarnings("ignore")


@dataclass
class Config81Y:
    n_trials: int = 30
    norm_trials: int = 3
    v_lines: tuple = (3.5, 21.5)
    train_trials = np.arange(4, 22).tolist()
    post_trials = np.arange(22, 30).tolist()
    output_folder: str = 'figures_81Y'
    condition: tuple = ('Uncertainty', ['Low', 'Med', 'High'])


@dataclass
class Config83Y:
    n_trials: int = 36
    norm_trials: int = 4
    v_lines: tuple = (3.5, 21.5)
    train_trials = np.arange(4, 28).tolist()
    post_trials = np.arange(28, 36).tolist()
    output_folder: str = 'figures_83Y'
    condition: tuple = ('Stimulation', ['20Hz', '80Hz', 'Sham'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folder", type=str, default='./')
    parser.add_argument("-n", "--name", type=str, default="figure.png")
    parser.add_argument("-p", "--plot", type=str, default="line_plot")
    parser.add_argument('-norm', '--normalization', type=bool, default=False)
    args = parser.parse_args()

    if args.folder[-3:] == '81Y':
        config = Config81Y()
    elif args.folder[-3:] == '83Y':
        config = Config83Y()

    files = glob.glob(f"{args.folder}/o*.csv")
    df = pd.DataFrame()
    for csv in files:
        file_name = re.split('/', string=csv)[-1]
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
            condition_ = config.condition[1][0]  # 20 Hz
        elif (condition == 2) or (condition == 5):
            condition_ = config.condition[1][1]  # 80 Hz
        elif (condition == 3) or (condition == 6):
            condition_ = config.condition[1][2]  # Sham

        tmp_df = pd.read_csv(csv)
        tmp_df['Participant'] = participant
        tmp_df['Reinforcement'] = reinforcement
        tmp_df[f"{config.condition[0]}"] = condition_
        tmp_df['NormTargetFeedback'] = (tmp_df['TargetFeedback'] /
                                        tmp_df['TargetFeedback'].iloc[
                                        :config.norm_trials].mean() * 100)
        tmp_df['NormTargetFeedforward'] = (tmp_df['TargetFeedforward'] /
                                           tmp_df['TargetFeedforward'].iloc[
                                           :config.norm_trials].mean() * 100)
        tmp_df['NormTargetTotalInfo'] = (tmp_df['TargetTotalInfo'] /
                                         tmp_df['TargetTotalInfo'].iloc[
                                         :config.norm_trials].mean() * 100)
        tmp_df['Condition'] = condition

        df = pd.concat((df, tmp_df))

    if args.plot == 'line_plot':
        line_plot(df, name=args.name, config=config,
                  normalization=args.normalization)
    elif args.plot == 'box_plot':
        box_plot(df, name=args.name, config=config)
    return


def box_plot(df, config, name):
    # breakpoint()
    df_train = df[(df['TrialNumber']).isin(config.train_trials)]
    df_post = df[(df['TrialNumber']).isin(config.post_trials)]

    # Train
    # breakpoint()
    df_train = df_train[['NormTargetFeedback', 'NormTargetFeedforward',
                         'NormTargetTotalInfo',
                         'Participant', 'Condition']].groupby(
        ['Condition', 'Participant']).mean()
    df_train = df_train.reset_index()
    df_post = df_post[['NormTargetFeedback', 'NormTargetFeedforward',
                       'NormTargetTotalInfo',
                       'Participant', 'Condition']].groupby(
        ['Condition', 'Participant']).mean()
    df_post = df_post.reset_index()

    # breakpoint()

    with plt.style.context('seaborn-v0_8-paper'):
        sns.set_context('talk')
        fig, ax = plt.subplots(3, 2, figsize=(12, 12),
                               sharex='all', sharey='row')

        sns.boxplot(df_train,
                    y="NormTargetFeedback", x="Condition",
                    ax=ax[0, 0], order=[1, 4, 2, 5, 3, 6])
        sns.boxplot(df_train,
                    y="NormTargetFeedforward", x="Condition",
                    ax=ax[1, 0], order=[1, 4, 2, 5, 3, 6])
        sns.boxplot(df_train,
                    y="NormTargetTotalInfo", x="Condition",
                    ax=ax[2, 0], order=[1, 4, 2, 5, 3, 6])
        ax[0, 0].set_title('Training')

        sns.boxplot(df_post,
                    y="NormTargetFeedback", x="Condition",
                    ax=ax[0, 1], order=[1, 4, 2, 5, 3, 6])
        sns.boxplot(df_post,
                    y="NormTargetFeedforward", x="Condition",
                    ax=ax[1, 1], order=[1, 4, 2, 5, 3, 6])
        sns.boxplot(df_post,
                    y="NormTargetTotalInfo", x="Condition",
                    ax=ax[2, 1], order=[1, 4, 2, 5, 3, 6])
        ax[0, 1].set_title('Post-Training')

        ax[2, 0].set_xticklabels([f"ReinOFF-{config.condition[1][0]}",
                                  f"ReinON-{config.condition[1][0]}",
                                  f"ReinOFF-{config.condition[1][1]}",
                                  f"ReinON-{config.condition[1][1]}",
                                  f"ReinOFF-{config.condition[1][2]}",
                                  f"ReinON-{config.condition[1][2]}"],
                                 rotation=45)
        ax[2, 1].set_xticklabels([f"ReinOFF-{config.condition[1][0]}",
                                  f"ReinON-{config.condition[1][0]}",
                                  f"ReinOFF-{config.condition[1][1]}",
                                  f"ReinON-{config.condition[1][1]}",
                                  f"ReinOFF-{config.condition[1][2]}",
                                  f"ReinON-{config.condition[1][2]}"],
                                 rotation=45)
        plt.tight_layout()
        plt.savefig(f"./{config.output_folder}/boxplot/{name}")


def line_plot(df, config, name, normalization):
    if normalization:
        var = 'Norm'
    else:
        var = ''

    dfs = []
    for rein in ['OFF', 'ON']:
        for condition_ in config.condition[1]:
            df_tmp = df[(df['Reinforcement'] == rein)
                        & (df[f"{config.condition[0]}"] == condition_)]
            dfs.append(df_tmp[[f'{var}TargetFeedback',
                               f'{var}TargetFeedforward',
                               f'{var}TargetTotalInfo']].groupby(
                level=0))
    with plt.style.context('seaborn-v0_8-paper'):
        sns.set_context('talk')
        fig, ax = plt.subplots(3, 3, figsize=(16, 8),
                               sharex='all', sharey='col')
        for i, condition_ in zip(range(3), config.condition[1]):
            ax[0, 0].set_title(f'{var}TargetFeedback')
            ax[i, 0].plot(dfs[0].mean().index + 1,
                          dfs[i][f'{var}TargetFeedback'].mean(), 'b')
            ax[i, 0].plot(dfs[0].mean().index + 1,
                          dfs[i + 3][f'{var}TargetFeedback'].mean(), 'r')

            ax[i, 0].fill_between(
                dfs[i][f'{var}TargetFeedback'].mean().index + 1,
                dfs[i][f'{var}TargetFeedback'].mean() -
                (dfs[i][f'{var}TargetFeedback'].std() / np.sqrt(24)),
                dfs[i][f'{var}TargetFeedback'].mean() +
                (dfs[i][f'{var}TargetFeedback'].std() /
                 np.sqrt(24)),
                color='b',
                alpha=0.2)

            ax[i, 0].fill_between(
                dfs[i + 3][f'{var}TargetFeedback'].mean().index + 1,
                dfs[i + 3][f'{var}TargetFeedback'].mean() -
                (dfs[0][f'{var}TargetFeedback'].std() / np.sqrt(24)),
                dfs[i + 3][f'{var}TargetFeedback'].mean() +
                ( dfs[i + 3][f'{var}TargetFeedback'].std() /
                  np.sqrt(24)),
                color='r',
                alpha=0.2)
            ax[i, 0].set_ylabel(condition_)
            ax[i, 0].set_xlim([1, config.n_trials])
            ax[i, 0].axvline(x=config.v_lines[0], color='k', linestyle='--')
            ax[i, 0].axvline(x=config.v_lines[1], color='k', linestyle='--')
            # ax[-1, 0].legend(['ReinOFF', 'ReinON'])

        for i, condition_ in zip(range(3), config.condition[1]):
            ax[0, 1].set_title(f'{var}TargetFeedforward')
            ax[i, 1].plot(dfs[0].mean().index + 1,
                          dfs[i][f'{var}TargetFeedforward'].mean(), 'b')
            ax[i, 1].plot(dfs[0].mean().index + 1,
                          dfs[i + 3][f'{var}TargetFeedforward'].mean(), 'r')

            ax[i, 1].fill_between(
                dfs[i][f'{var}TargetFeedforward'].mean().index + 1,
                dfs[i][f'{var}TargetFeedforward'].mean() -
                (dfs[i][f'{var}TargetFeedforward'].std() / np.sqrt(24)),
                dfs[i][f'{var}TargetFeedforward'].mean() + (
                        dfs[i][f'{var}TargetFeedforward'].std()
                        / np.sqrt(24)), color='b',
                alpha=0.2)

            ax[i, 1].fill_between(
                dfs[i + 3][f'{var}TargetFeedforward'].mean().index + 1,
                dfs[i + 3][f'{var}TargetFeedforward'].mean() -
                (dfs[0][f'{var}TargetFeedforward'].std() / np.sqrt(24)),
                dfs[i + 3][f'{var}TargetFeedforward'].mean() +
                (dfs[i + 3][f'{var}TargetFeedforward'].std() /
                 np.sqrt(24)),
                color='r',
                alpha=0.2)
            ax[i, 1].set_ylabel(condition_)
            ax[i, 1].axvline(x=config.v_lines[0], color='k', linestyle='--')
            ax[i, 1].axvline(x=config.v_lines[1], color='k', linestyle='--')

        for i, condition_ in zip(range(3), config.condition[1]):
            ax[0, 2].set_title(f'{var}TargetTotalInfo')
            ax[i, 2].plot(dfs[0].mean().index + 1,
                          dfs[i][f'{var}TargetTotalInfo'].mean(), 'b')
            ax[i, 2].plot(dfs[0].mean().index + 1,
                          dfs[i + 3][f'{var}TargetTotalInfo'].mean(), 'r')

            ax[i, 2].fill_between(
                dfs[i][f'{var}TargetTotalInfo'].mean().index + 1,
                dfs[i][f'{var}TargetTotalInfo'].mean() -
                (dfs[i][f'{var}TargetTotalInfo'].std() / np.sqrt(24)),
                dfs[i][f'{var}TargetTotalInfo'].mean() +
                (dfs[i][f'{var}TargetTotalInfo'].std() / np.sqrt(24)),
                color='b',
                alpha=0.2)

            ax[i, 2].fill_between(
                dfs[i + 3][f'{var}TargetTotalInfo'].mean().index + 1,
                dfs[i + 3][f'{var}TargetTotalInfo'].mean() -
                (dfs[0][f'{var}TargetTotalInfo'].std() / np.sqrt(24)),
                dfs[i + 3][f'{var}TargetTotalInfo'].mean() +
                (dfs[i + 3][f'{var}TargetTotalInfo'].std() /
                 np.sqrt(24)),
                color='r',
                alpha=0.2)
            ax[i, 2].set_ylabel(condition_)
            ax[-1, 2].legend(['ReinOFF', 'ReinON'])
            ax[i, 2].axvline(x=config.v_lines[0], color='k', linestyle='--')
            ax[i, 2].axvline(x=config.v_lines[1], color='k', linestyle='--')

        plt.tight_layout()
        plt.savefig(f"./{config.output_folder}/line_plot/{name}")
        return


if __name__ == '__main__':
    main()
