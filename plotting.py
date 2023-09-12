import argparse
import os
import re

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import glob
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folder", type=str, default='./')
    parser.add_argument("-c", "--cond", type=int, default=None)
    parser.add_argument("-v", "--variable", type=str, default=None)
    args = parser.parse_args()

    files = glob.glob(f"{args.folder}/*/*.csv")
    df = pd.DataFrame()
    for csv in files:
        # breakpoint()
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
            uncertainty = 'Low'
        elif (condition == 2) or (condition == 5):
            uncertainty = 'Medium'
        elif (condition == 3) or (condition == 6):
            uncertainty = 'High'

        tmp_df = pd.read_csv(csv)
        tmp_df['Participant'] = participant
        tmp_df['Reinforcement'] = reinforcement
        tmp_df['Uncertainty'] = uncertainty

        df = pd.concat((df, tmp_df))
    # breakpoint()
    plot_info_transfer(df, cond=args.cond, variable=args.variable)
    return


def plot_info_transfer(df, cond=None, variable=None):
    if cond is not None:
        dfmean = df.groupby(level=0).mean()
        dfstd = df.groupby(level=0).std()
        t = dfmean.TrialNumber + 6  # adding 6 to trial number

        with plt.style.context('seaborn-v0_8-paper'):
            sns.set_context('talk')
            fig, ax = plt.subplots(1, 2, figsize=(12, 4),
                                   sharex='all')
            # breakpoint()
            fb_target_mean = dfmean.TargetFeedback
            fb_target_std = dfstd.TargetFeedback / np.sqrt(24)
            # fb_colour_mean = dfmean.ColourFeedback
            # fb_colour_std = dfstd.ColourFeedback / np.sqrt(24)
            # fb_both_mean = dfmean.BothFeedback
            # fb_both_std = dfstd.BothFeedback / np.sqrt(24)

            ff_target_mean = dfmean.TargetFeedforward
            ff_target_std = dfstd.TargetFeedforward / np.sqrt(24)
            # ff_colour_mean = dfmean.ColourFeedforward
            # ff_colour_std = dfstd.ColourFeedforward / np.sqrt(24)
            # ff_both_mean = dfmean.BothFeedforward
            # ff_both_std = dfstd.BothFeedforward / np.sqrt(24)

            ax[0].plot(t, dfmean.TargetFeedback, 'b')  # , t,
            # dfmean.ColourFeedback, 'g', t, dfmean.BothFeedback,
            # 'r')
            ax[0].fill_between(t, fb_target_mean - fb_target_std,
                               fb_target_mean + fb_target_std,
                               color='b', alpha=0.2)

            # ax[0].fill_between(t, fb_colour_mean - fb_colour_std,
            #                    fb_colour_mean + fb_colour_std,
            #                    color='g', alpha=0.2)
            #
            # ax[0].fill_between(t, fb_both_mean - fb_both_std,
            #                    fb_both_mean + fb_both_std,
            #                    color='r', alpha=0.2)
            ax[0].set_title(f"Cond: {cond}. Feedback")
            ax[0].set_xlabel("Trial number")
            ax[0].set_ylabel("FB")
            ax[0].set_ylim([0.06, 0.09])
            # ax[0].legend(['Target', 'Colour', 'Both'])
            # ax[0].set_ylim([0, 0.1])
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
            for uncertainty in ['Low', 'Medium', 'High']:
                df_tmp = df[(df['Reinforcement'] == rein)
                            & (df['Uncertainty'] == uncertainty)]
                # breakpoint()
                dfs.append(df_tmp[['TargetFeedback', 'TargetFeedforward']].groupby(level=0))

        # breakpoint()
        with plt.style.context('seaborn-v0_8-paper'):
            sns.set_context('talk')
            fig, ax = plt.subplots(3, 2, figsize=(12, 8),
                                   sharex='all', sharey='col')
            for i, uncertainty in zip(range(3), ['Low', 'Medium', 'High']):
                ax[0,0].set_title('TargetFeedback')
                ax[i, 0].plot(dfs[0].mean().index + 6,
                              dfs[i]['TargetFeedback'].mean(), 'b')
                ax[i, 0].plot(dfs[0].mean().index + 6,
                              dfs[i + 3]['TargetFeedback'].mean(), 'r')

                ax[i, 0].fill_between(dfs[i]['TargetFeedback'].mean().index + 6,
                                      dfs[i]['TargetFeedback'].mean() - (dfs[
                    i]['TargetFeedback'].std() / np.sqrt(24)),
                                      dfs[i]['TargetFeedback'].mean() + (
                                          dfs[i]['TargetFeedback'].std() /
                                          np.sqrt(24)),
                                      color='b',
                                      alpha=0.2)

                ax[i, 0].fill_between(dfs[i+3]['TargetFeedback'].mean().index + 6, dfs[i+3]['TargetFeedback'].mean() -
                                      (dfs[0]['TargetFeedback'].std() /
                                       np.sqrt(24)),
                                      dfs[i+3]['TargetFeedback'].mean() + (
                                          dfs[i+3]['TargetFeedback'].std() /
                                          np.sqrt(24)),
                                      color='r',
                                      alpha=0.2)
                ax[i,0].set_ylabel(uncertainty)
                ax[i, 0].axvline(x=9, color='k', linestyle='--')
                ax[i, 0].axvline(x=27, color='k', linestyle='--')
                # ax[-1, 0].legend(['ReinOFF', 'ReinON'])

            for i, uncertainty in zip(range(3), ['Low', 'Medium', 'High']):
                ax[0,1].set_title('TargetFeedforward')
                ax[i, 1].plot(dfs[0].mean().index + 6,
                              dfs[i]['TargetFeedforward'].mean(), 'b')
                ax[i, 1].plot(dfs[0].mean().index + 6,
                              dfs[i + 3]['TargetFeedforward'].mean(), 'r')

                ax[i, 1].fill_between(dfs[i]['TargetFeedforward'].mean().index + 6,
                                      dfs[i]['TargetFeedforward'].mean() - (dfs[
                    i]['TargetFeedforward'].std() / np.sqrt(24)),
                                      dfs[i]['TargetFeedforward'].mean() + (
                                          dfs[i]['TargetFeedforward'].std()
                                          / np.sqrt(24)), color='b',
                                      alpha=0.2)

                ax[i, 1].fill_between(dfs[i+3]['TargetFeedforward'].mean(

                ).index + 6, dfs[i+3]['TargetFeedforward'].mean() -
                                      (dfs[0]['TargetFeedforward'].std() /
                                       np.sqrt(24)),
                                      dfs[i+3]['TargetFeedforward'].mean() +
                                      (dfs[i+3]['TargetFeedforward'].std() /
                                       np.sqrt(24)),
                                      color='r',
                                      alpha=0.2)
                ax[i,1].set_ylabel(uncertainty)
                ax[-1, 1].legend(['ReinOFF', 'ReinON'])
                ax[i, 1].axvline(x=9, color='k', linestyle='--')
                ax[i, 1].axvline(x=27, color='k', linestyle='--')


            plt.tight_layout()
            plt.savefig(f"./figures/overlapped/test.png")
        return


if __name__ == '__main__':
    main()
