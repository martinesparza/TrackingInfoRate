# Import classes
import pathlib

import matplotlib.pyplot as plt
import information_transfer as it
import numpy as np
import pandas as pd
import argparse
import os
from __init__ import PATHS
from tqdm import tqdm
import seaborn as sns
from joblib import Parallel, delayed
from plotting import Config81Y, Config83Y, ConfigEMG
from operator import itemgetter
from itertools import groupby

import warnings

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--filename", type=str, default='')
parser.add_argument("-p", "--path", type=str, default='data/')
parser.add_argument("-pl", "--plot", type=bool, default=False)
parser.add_argument("-exp", "--experiment", type=str, default='emg')
args = parser.parse_args()


# functions for dealing with nans
def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]


def interpolate_nans(y):
    nans, x = nan_helper(y)
    y[nans] = np.interp(x(nans), x(~nans), y[~nans])
    return y


def group_consecutive(data):
    groups = []
    for k, g in groupby(enumerate(data), lambda x: x[0] - x[1]):
        groups.append(list(map(itemgetter(1), g)))
    return groups


# load data
def load_data(path, filename, diagnostics, project='emg'):
    csv = np.genfromtxt(path + filename, delimiter=",")
    if project == '81Y':
        cursor = csv[1:418, 0:35]
        colour = csv[1:418, 35:70]
        target = csv[1:418, 70:105]
        allTrials = list(range(35))

    elif project == '83Y':
        cursor = csv[1:418, 0:36]
        colour = csv[1:418, 36:72]
        target = csv[1:418, 72:108]
        allTrials = list(range(36))

    elif project == '81Y_titr':
        cursor = csv[1:418, 0:42]
        colour = csv[1:418, 42:84]
        target = csv[1:418, 84:126]
        allTrials = list(range(42))

    elif project == 'emg':
        cursor = csv[1:358, 0:29]
        colour = csv[1:358, 29:58]
        target = csv[1:358, 58:87]
        allTrials = list(range(29))
        cond = filename[-5:-4]
        block = filename[-11:-10]
        if len(filename) == 24:
            subj = filename[5:7]
        else:
            subj = filename[5:6]

    # remove nans
    nanTrials = []
    for idx, tr in enumerate(cursor.T):
        remove_trial = False
        if np.all(np.isnan(tr)):
            nanTrials.append(idx)

        elif np.any(np.isnan(tr)):

            groups = group_consecutive(np.where(np.isnan(tr))[0])
            for group in groups:
                if len(group) > 10:
                    nanTrials.append(idx)
                    remove_trial = True
                    break

            if not remove_trial:
                tr = interpolate_nans(tr)
                cursor[:, idx] = tr

    if nanTrials:
        for trial_ in nanTrials:
            diagnostics.loc[-1] = [subj, block, cond, trial_]
            diagnostics = diagnostics.sort_index().reset_index(drop=True)

    allTrials = list(set(allTrials) - set(nanTrials))
    cursor = np.delete(cursor, nanTrials, 1)
    colour = np.delete(colour, nanTrials, 1)
    target = np.delete(target, nanTrials, 1)
    # breakpoint()

    return cursor, colour, target, allTrials, diagnostics


def compute_info_transfer(path=None, filename=None, exp=None, groupby='',
                          diagnostics=None):
    # breakpoint()
    if exp == 'emg':
        config = ConfigEMG

    if filename is None:
        if (args.path):
            df = pd.DataFrame()
            directory = os.fsencode(args.path)
            for file in tqdm(os.listdir(directory)):
                filename = os.fsdecode(file)
                # cond = int(filename[-5:-4])
                if filename.endswith(".csv"):
                    new_df, diagnostics = compute_info_transfer(args.path,
                                                                filename,
                                                                exp=exp,
                                                                diagnostics=diagnostics)
                    # new_df['Cond'] = cond * np.ones(len(new_df)).astype(
                    #     'int32')
                    df = pd.concat((df, new_df))
            diagnostics.to_csv(f"./output_emg_v2/diagnostics.csv",
                               index=False)

            dfmean = df.groupby(level=0).mean()
            dfstd = df.groupby(level=0).std()

            # plotting average
            if groupby == 'subject':
                t = dfmean.TrialNumber  # adding 6 to trial number because of the 6 training trials

                sns.set_context('talk')
                fig, ax = plt.subplots(1, 2, figsize=(12, 4),
                                       sharex='all', sharey='all')
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
                ax[0].set_title("Feedback info")
                ax[0].set_xlabel("Trial number")
                ax[0].set_ylabel("FB")
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
                ax[1].set_title("Feedforward info")
                ax[1].set_xlabel("Trial number")
                ax[1].set_ylabel("FF")
                ax[1].set_xlim([0, 30])
                # plt.legend(['Target', 'Colour', 'Both'])
                plt.tight_layout()
                # plt.suptitle(f"Condition: {filename[-5:-4]}")
                # plt.savefig(f"./figures/only_ff/cond{filename[-5:-4]}")

            elif groupby == 'ff_fb':
                sns.set_context('notebook')
                fig, ax = plt.subplots(2, 3, figsize=(12, 6),
                                       sharex='all', sharey='row')
                df2 = pd.DataFrame()
                for cond in range(1, 7):
                    df_cond = df[df["Cond"] == cond]
                    df2 = pd.concat((df2, df_cond.groupby(level=0).mean()))
                # breakpoint()

                use_trials = np.arange(3, 21).tolist()
                df = df[(df.index).isin(use_trials)]
                # Feedback
                sns.boxplot(df,
                            y="TargetFeedback", x="Cond",
                            ax=ax[0, 0], order=[1, 4, 2, 5, 3, 6])
                sns.boxplot(df,
                            y="ColourFeedback", x="Cond",
                            ax=ax[0, 1], order=[1, 4, 2, 5, 3, 6])
                sns.boxplot(df,
                            y="BothFeedback", x="Cond",
                            ax=ax[0, 2], order=[1, 4, 2, 5, 3, 6])

                # Feedforward
                sns.boxplot(df,
                            y="TargetFeedforward", x="Cond",
                            ax=ax[1, 0], order=[1, 4, 2, 5, 3, 6])
                sns.boxplot(df,
                            y="ColourFeedforward", x="Cond",
                            ax=ax[1, 1], order=[1, 4, 2, 5, 3, 6])
                sns.boxplot(df,
                            y="BothFeedforward", x="Cond",
                            ax=ax[1, 2], order=[1, 4, 2, 5, 3, 6])
                ax[1, 2].set_xticklabels(['ReinOFF-Low', 'ReinON-Low',
                                          'ReinOFF-Med', 'ReinON-Med',
                                          'ReinOFF-High', 'ReinOFF-High'],
                                         rotation=45)
                ax[1, 1].set_xticklabels(['ReinOFF-Low', 'ReinON-Low',
                                          'ReinOFF-Med', 'ReinON-Med',
                                          'ReinOFF-High', 'ReinOFF-High'],
                                         rotation=45)
                ax[1, 0].set_xticklabels(['ReinOFF-Low', 'ReinON-Low',
                                          'ReinOFF-Med', 'ReinON-Med',
                                          'ReinOFF-High', 'ReinON-High'],
                                         rotation=45)

                plt.tight_layout()
                plt.savefig(f"./figures/groupby_FB_FF/test.png")


    else:
        cursor, colour, target, allTrials, diagnostics = load_data(path,
                                                                   filename,
                                                                   diagnostics)
        # only target
        targetFB = []
        targetINFO = []
        order = [4, 3]

        for col in range(cursor.shape[1]):
            try:
                targetFB.append(
                    it.compute_FB([target[:, col]], cursor[:, col], order,
                                  VMD))
                targetINFO.append(
                    it.compute_total_info([target[:, col]], cursor[:, col],
                                          order,
                                          VMD))
            except:
                breakpoint()
        targetFF = np.array(targetINFO) - np.array(targetFB)
        # only colour
        colourFB = []
        colourINFO = []
        order = [0, 3]
        for col in range(colour.shape[1]):
            colourFB.append(
                it.compute_FB([colour[:, col]], cursor[:, col], order, VMD))
            colourINFO.append(
                it.compute_total_info([colour[:, col]], cursor[:, col], order,
                                      VMD))
        colourFF = np.array(colourINFO) - np.array(colourFB)

        # both cursor and colour
        bothFB = []
        bothINFO = []
        order = [4, 0, 3]

        for col in range(cursor.shape[1]):
            bothFB.append(
                it.compute_FB([target[:, col], colour[:, col]], cursor[:, col],
                              order, VMD))
            bothINFO.append(
                it.compute_total_info([target[:, col], colour[:, col]],
                                      cursor[:, col], order, VMD))
        bothFF = np.array(bothINFO) - np.array(bothFB)

        # saving
        zipped = list(
            zip(allTrials, targetFB, targetFF, targetINFO, colourFB, colourFF,
                colourINFO, bothFB, bothFF, bothINFO))
        df = pd.DataFrame(zipped, columns=['TrialNumber', 'TargetFeedback',
                                           'TargetFeedforward',
                                           'TargetTotalInfo',
                                           'ColourFeedback',
                                           'ColourFeedforward',
                                           'ColourTotalInfo',
                                           'BothFeedback', 'BothFeedforward',
                                           'BothTotalInfo'])
        df['TrialNumber'] = df['TrialNumber'] + 1
        # if int(df.iloc[-1]['TrialNumber']) != 42:
        #     df.loc[len(df)] = [int(42),
        #                        np.nan, np.nan, np.nan, np.nan, np.nan,
        #                        np.nan, np.nan, np.nan, np.nan]
        #
        for trial in np.arange(1, config.n_trials + 1):
            if len(df) > trial - 1:
                if int(df.loc[trial - 1]['TrialNumber']) != trial:
                    df.loc[trial - 1 - 0.5] = [trial,
                                               np.nan, np.nan, np.nan, np.nan,
                                               np.nan,
                                               np.nan, np.nan, np.nan, np.nan]

                    df = df.sort_index().reset_index(drop=True)
            else:
                df.loc[trial - 1] = ([trial,
                                      np.nan, np.nan, np.nan, np.nan,
                                      np.nan,
                                      np.nan, np.nan, np.nan, np.nan])

                df = df.sort_index().reset_index(drop=True)

        df['TrialNumber'] = df['TrialNumber'].astype('int')
        df.to_csv(f"output_emg_v2/output_{filename}",
                  index=False)

        # plotting
        if args.plot:
            t = np.array(
                allTrials) + 6  # adding 6 to trial number because of the 6 training trials
            fig = plt.figure()
            plt.subplot(1, 2, 1)
            plt.plot(t, targetFB, 'b', t, colourFB, 'g', t, bothFB, 'r')
            plt.title("Feedback info")
            plt.xlabel("Trial number")
            plt.ylabel("FB")
            plt.subplot(1, 2, 2)
            plt.plot(t, targetFF, 'b', t, colourFF, 'g', t, bothFF, 'r')
            plt.title("Feedforward info")
            plt.xlabel("Trial number")
            plt.ylabel("FF")
            plt.legend(['Target', 'Colour', 'Both'])
            plt.suptitle(f"File: {args.filename}")
            plt.savefig(f"./figures/{args.filename}.png")

        return df, diagnostics


VMD = 15
if args.filename:
    compute_info_transfer(args.path, args.filename)
else:
    diagnostics = pd.DataFrame(columns=['Subject', 'Block', 'Condition',
                                        'RemovedTrial'])
    compute_info_transfer(exp=args.experiment, diagnostics=diagnostics)
