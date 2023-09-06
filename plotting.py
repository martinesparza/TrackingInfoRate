import argparse
import os

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import glob
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folder", type=str, default='./')
    parser.add_argument("-c", "--cond", type=int, default=1)
    args = parser.parse_args()

    csvs = glob.glob(f"{args.folder}/*")
    df = pd.DataFrame()
    for csv in csvs:
        df = pd.concat((df, pd.read_csv(csv)))

    plot_info_transfer(df, cond=args.cond)
    return


def plot_info_transfer(df, cond=None):
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

        return


if __name__ == '__main__':
    main()
