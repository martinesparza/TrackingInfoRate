# Import classes
import matplotlib.pyplot as plt
import information_transfer as it
import numpy as np
import pandas as pd
import argparse
import os
from tqdm import tqdm
from plotting import Config81Y, Config83Y, ConfigEMG
from plotting_titr import Config81Y_titr
from operator import itemgetter
from itertools import groupby

import warnings

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("-in_path", "--input_path", type=str, default='')
parser.add_argument("-out_path", "--out_path", type=str, default='')
parser.add_argument("-c", "--config", type=str, default='')
parser.add_argument("-vmd", "--vmd", type=int, default=15)


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
def load_data(path, filename, diagnostics, config):
    csv = np.genfromtxt(path + filename, delimiter=",")
    cursor = csv[1:418, 0:config.n_trials]
    colour = csv[1:418, config.n_trials:int(config.n_trials * 2)]
    target = csv[1:418, int(config.n_trials * 2):int(config.n_trials * 3)]
    allTrials = list(range(config.n_trials))

    if (config.__name__ == 'Config81Y') or (config.__name__ == 'Config83Y'):
        if len(filename) == 21:  #
            subj = int(filename[3:4])
        else:
            subj = int(filename[3:5])

    if config.__name__ == 'Config81Y_titr':
        if len(filename) == 15:  #
            subj = int(filename[3:4])
        else:
            subj = int(filename[3:5])

    elif config.__name__ == 'ConfigEMG':
        if len(filename) == 23:  #
            subj = int(filename[5:6])
        else:
            subj = int(filename[5:7])


    cond = filename[-5:-4]
    block = filename[-11:-10]

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


def compute_info_transfer(path=None, filename=None, config=None,
                          diagnostics=None):
    if filename is None:
        df = pd.DataFrame()
        directory = os.fsencode(args.input_path)
        for file in tqdm(os.listdir(directory)):
            filename = os.fsdecode(file)
            if filename.endswith(".csv"):
                new_df, diagnostics = compute_info_transfer(args.input_path,
                                                            filename,
                                                            config=config,
                                                            diagnostics=diagnostics)

                df = pd.concat((df, new_df))
        diagnostics.to_csv(f"{args.out_path}/diagnostics.csv",
                           index=False)

    else:
        cursor, colour, target, allTrials, diagnostics = load_data(path,
                                                                   filename,
                                                                   diagnostics,
                                                                   config
                                                                   )
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

        for trial in np.arange(1, config.n_trials + 1):
            if len(df) > trial - 1:
                if int(df.loc[trial - 1]['TrialNumber']) != trial:
                    df.loc[trial - 1 - 0.5] = [trial,
                                               np.nan, np.nan, np.nan,
                                               np.nan,
                                               np.nan,
                                               np.nan, np.nan, np.nan,
                                               np.nan]

                    df = df.sort_index().reset_index(drop=True)
            else:
                df.loc[trial - 1] = ([trial,
                                      np.nan, np.nan, np.nan, np.nan,
                                      np.nan,
                                      np.nan, np.nan, np.nan, np.nan])

                df = df.sort_index().reset_index(drop=True)

        df['TrialNumber'] = df['TrialNumber'].astype('int')
        df.to_csv(f"{args.out_path}/output_{filename}",
                  index=False)
        return df, diagnostics


def check_fields(args):
    if not args.input_path:
        raise ValueError('Missing input_path parameter')

    if not args.out_path:
        raise ValueError('Missing out_path parameter')


if __name__ == '__main__':
    args = parser.parse_args()
    check_fields(args)

    VMD = args.vmd
    if args.config == '81':
        config = Config81Y
    elif args.config == '81_titr':
        config = Config81Y_titr
    elif args.config == '83':
        config = Config83Y
    elif args.config == 'emg':
        config = ConfigEMG

    diagnostics = pd.DataFrame(columns=['Subject', 'Block', 'Condition',
                                        'RemovedTrial'])
    compute_info_transfer(config=config, diagnostics=diagnostics)
