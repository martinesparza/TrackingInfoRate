import argparse
import os
import re

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import glob
import numpy as np

import warnings

warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folder", type=str, default='./')
    parser.add_argument("-c", "--cond", type=int, default=None)
    parser.add_argument("-v", "--variable", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default="figure")
    parser.add_argument("-p", "--plot", type=str, default="line_plot")
    args = parser.parse_args()

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
            uncertainty = 'Low'  # 20 Hz
        elif (condition == 2) or (condition == 5):
            uncertainty = 'Med'  # 80 Hz
        elif (condition == 3) or (condition == 6):
            uncertainty = 'High'  # Sham

        tmp_df = pd.read_csv(csv)
        tmp_df['Participant'] = participant
        tmp_df['Reinforcement'] = reinforcement
        tmp_df['Uncertainty'] = uncertainty
        tmp_df['NormTargetFeedback'] = (tmp_df['TargetFeedback'] / tmp_df[
                                                                       'TargetFeedback'].iloc[
                                                                   :4].mean()
                                        * 100)
        tmp_df['NormTargetFeedforward'] = tmp_df['TargetFeedforward'] / tmp_df[
                                                                            'TargetFeedforward'].iloc[
                                                                        :4].mean() * 100
        tmp_df['NormTargetTotalInfo'] = tmp_df['TargetTotalInfo'] / tmp_df[
                                                                        'TargetTotalInfo'].iloc[
                                                                    :4].mean() * 100
        tmp_df['Condition'] = condition

        df = pd.concat((df, tmp_df))
    if args.plot == 'line_plot':
        line_plot(df, cond=args.cond, variable=args.variable,
                  name=args.name)
    elif args.plot == 'box_plot':
        box_plot(df, name=args.name)
    return


def box_plot(df, name):
    df_train = df[(df['TrialNumber']).isin(np.arange(5, 29).tolist())]  # 5
    # - 29
    df_post = df[(df['TrialNumber']).isin(np.arange(29, 36).tolist())]  # 29
    #  35

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

        ax[2, 0].set_xticklabels(['ReinOFF-20Hz', 'ReinON-20Hz',
                                  'ReinOFF-80Hz', 'ReinON-80Hz',
                                  'ReinOFF-Sham', 'ReinON-Sham'],
                                 rotation=45)
        ax[2, 1].set_xticklabels(['ReinOFF-20Hz', 'ReinON-20Hz',
                                  'ReinOFF-80Hz', 'ReinON-80Hz',
                                  'ReinOFF-Sham', 'ReinON-Sham'],
                                 rotation=45)
        plt.tight_layout()
        plt.savefig(f"./figures_83Y/boxplot/train_and_post.png")


def line_plot(df, cond=None, variable=None, name=None):

    if cond is not None:
        dfmean = df.groupby(level=0).mean()
        dfstd = df.groupby(level=0).std()
        t = dfmean.TrialNumber  # adding 6 to trial number

        with plt.style.context('seaborn-v0_8-paper'):
            sns.set_context('talk')
            fig, ax = plt.subplots(1, 2, figsize=(12, 4),
                                   sharex='all')
            fb_target_mean = dfmean.TargetFeedback
            fb_target_std = dfstd.TargetFeedback / np.sqrt(24)


            ff_target_mean = dfmean.TargetFeedforward
            ff_target_std = dfstd.TargetFeedforward / np.sqrt(24)


            ax[0].plot(t, dfmean.TargetFeedback, 'b')
            ax[0].fill_between(t, fb_target_mean - fb_target_std,
                               fb_target_mean + fb_target_std,
                               color='b', alpha=0.2)


            ax[0].set_title(f"Cond: {cond}. Feedback")
            ax[0].set_xlabel("Trial number")
            ax[0].set_ylabel("FB")
            ax[0].set_ylim([0.06, 0.09])

            ax[1].plot(t, dfmean.TargetFeedforward, 'b')  # , t,
            # dfmean.ColourFeedforward, 'g', t,
            # dfmean.BothFeedforward, 'r')
            ax[1].fill_between(t, ff_target_mean - ff_target_std,
                               ff_target_mean + ff_target_std,
                               color='b', alpha=0.2)

            # ax[1].fill_between(t, ff_colour_mean - ff_colour_std,
            #                    ff_colour_mean + ff_colour_std,
            #                    color='g', alpha=0.2)
            #
            # ax[1].fill_between(t, ff_both_mean - ff_both_std,
            #                    ff_both_mean + ff_both_std,
            #                    color='r', alpha=0.2)
            ax[1].set_title(f"Cond: {cond}. Feedforward")
            ax[1].set_xlabel("Trial number")
            ax[1].set_ylabel("FF")
            ax[1].set_xlim([6, 35])
            ax[1].set_ylim([1.4, 1.7])

            # plt.legend(['Target', 'Colour', 'Both'])
            plt.tight_layout()
            # plt.suptitle(f"Condition: {filename[-5:-4]}")
            plt.savefig(f"./figures/only_ff/cond{cond}")
    else:
        dfs = []
        for rein in ['OFF', 'ON']:
            for uncertainty in ['Low', 'Med', 'High']:
                df_tmp = df[(df['Reinforcement'] == rein)
                            & (df['Uncertainty'] == uncertainty)]
                dfs.append(df_tmp[['NormTargetFeedback',
                                   'NormTargetFeedforward',
                                   'NormTargetTotalInfo']].groupby(
                    level=0))

        with plt.style.context('seaborn-v0_8-paper'):
            sns.set_context('talk')
            fig, ax = plt.subplots(3, 3, figsize=(16, 8),
                                   sharex='all', sharey='col')
            for i, uncertainty in zip(range(3), ['Low', 'Med', 'High']):
                ax[0, 0].set_title('NormTargetFeedback')
                ax[i, 0].plot(dfs[0].mean().index,
                              dfs[i]['NormTargetFeedback'].mean(), 'b')
                ax[i, 0].plot(dfs[0].mean().index,
                              dfs[i + 3]['NormTargetFeedback'].mean(), 'r')

                ax[i, 0].fill_between(dfs[i]['NormTargetFeedback'].mean(

                ).index,
                                      dfs[i]['NormTargetFeedback'].mean() - (
                                              dfs[
                                                  i][
                                                  'NormTargetFeedback'].std() /
                                              np.sqrt(
                                          24)),
                                      dfs[i]['NormTargetFeedback'].mean() + (
                                              dfs[i][
                                                  'NormTargetFeedback'].std() /
                                              np.sqrt(24)),
                                      color='b',
                                      alpha=0.2)

                ax[i, 0].fill_between(dfs[i + 3]['NormTargetFeedback'].mean(

                ).index,
                                      dfs[i + 3]['NormTargetFeedback'].mean() -
                                      (dfs[0]['NormTargetFeedback'].std() /
                                       np.sqrt(24)),
                                      dfs[i + 3][
                                          'NormTargetFeedback'].mean() + (
                                              dfs[i + 3][
                                                  'NormTargetFeedback'].std() /
                                              np.sqrt(24)),
                                      color='r',
                                      alpha=0.2)
                ax[i, 0].set_ylabel(uncertainty)
                ax[i, 0].axvline(x=4.5, color='k', linestyle='--')
                ax[i, 0].axvline(x=28.5, color='k', linestyle='--')
                # ax[-1, 0].legend(['ReinOFF', 'ReinON'])

            for i, uncertainty in zip(range(3), ['Low', 'Med', 'High']):
                ax[0, 1].set_title('NormTargetFeedforward')
                ax[i, 1].plot(dfs[0].mean().index,
                              dfs[i]['NormTargetFeedforward'].mean(), 'b')
                ax[i, 1].plot(dfs[0].mean().index,
                              dfs[i + 3]['NormTargetFeedforward'].mean(), 'r')

                ax[i, 1].fill_between(
                    dfs[i]['NormTargetFeedforward'].mean().index,
                    dfs[i]['NormTargetFeedforward'].mean() - (dfs[
                                                                  i][
                                                                  'NormTargetFeedforward'].std() / np.sqrt(
                        24)),
                    dfs[i]['NormTargetFeedforward'].mean() + (
                            dfs[i]['NormTargetFeedforward'].std()
                            / np.sqrt(24)), color='b',
                    alpha=0.2)

                ax[i, 1].fill_between(dfs[i + 3]['NormTargetFeedforward'].mean(

                ).index, dfs[i + 3]['NormTargetFeedforward'].mean() -
                                      (dfs[0]['NormTargetFeedforward'].std() /
                                       np.sqrt(24)),
                                      dfs[i + 3][
                                          'NormTargetFeedforward'].mean() +
                                      (dfs[i + 3][
                                           'NormTargetFeedforward'].std() /
                                       np.sqrt(24)),
                                      color='r',
                                      alpha=0.2)
                ax[i, 1].set_ylabel(uncertainty)
                ax[i, 1].axvline(x=4.5, color='k', linestyle='--')
                ax[i, 1].axvline(x=28.5, color='k', linestyle='--')

            for i, uncertainty in zip(range(3), ['Low', 'Med', 'High']):
                ax[0, 2].set_title('NormTargetTotalInfo')
                ax[i, 2].plot(dfs[0].mean().index,
                              dfs[i]['NormTargetTotalInfo'].mean(), 'b')
                ax[i, 2].plot(dfs[0].mean().index,
                              dfs[i + 3]['NormTargetTotalInfo'].mean(), 'r')

                ax[i, 2].fill_between(
                    dfs[i]['NormTargetTotalInfo'].mean().index,
                    dfs[i]['NormTargetTotalInfo'].mean() - (dfs[
                                                                  i][
                                                                  'NormTargetTotalInfo'].std()
                                          / np.sqrt(
                        24)),
                    dfs[i]['NormTargetTotalInfo'].mean() + (
                            dfs[i]['NormTargetTotalInfo'].std()
                            / np.sqrt(24)), color='b',
                    alpha=0.2)

                ax[i, 2].fill_between(dfs[i + 3]['NormTargetTotalInfo'].mean(

                ).index,
                                      dfs[i + 3][
                                          'NormTargetTotalInfo'].mean() -
                                      (dfs[0]['NormTargetTotalInfo'].std() /
                                       np.sqrt(24)),
                                      dfs[i + 3][
                                          'NormTargetTotalInfo'].mean() +
                                      (dfs[i + 3][
                                           'NormTargetTotalInfo'].std() /
                                       np.sqrt(24)),
                                      color='r',
                                      alpha=0.2)
                ax[i, 2].set_ylabel(uncertainty)
                ax[-1, 2].legend(['ReinOFF', 'ReinON'])
                ax[i, 2].axvline(x=2.5, color='k', linestyle='--')
                ax[i, 2].axvline(x=28.5, color='k', linestyle='--')

            plt.tight_layout()
            plt.savefig(f"./figures_83Y/overlapped/{name}.png")
        return


if __name__ == '__main__':
    main()
