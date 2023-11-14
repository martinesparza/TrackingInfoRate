import argparse
import os
import re
from dataclasses import dataclass

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import glob
import numpy as np
import matplotlib.patches

import warnings

warnings.filterwarnings("ignore")


@dataclass
class Config81Y:
    n_trials: int = 35
    trials_to_plot: tuple = (6, 35)
    norm_trials = np.arange(6, 9)
    v_lines: tuple = (8.5, 26.5)
    train_trials = np.arange(9, 27).tolist()
    post_trials = np.arange(27, 36).tolist()
    output_folder: str = 'figures_81Y'
    condition: tuple = ('Uncertainty', ['Low', 'Med', 'High'])
    name_length: tuple = (28, 10, 11, 12)


@dataclass
class Config83Y:
    n_trials: int = 36
    trials_to_plot: tuple = (1, 36)
    norm_trials = np.arange(1, 5).tolist()
    v_lines: tuple = (4.5, 28.5)
    train_trials = np.arange(5, 29).tolist()
    post_trials = np.arange(29, 37).tolist()
    output_folder: str = 'figures_83Y'
    condition: tuple = ('Stimulation', ['20Hz', '80Hz', 'Sham'])
    name_length: tuple = (28, 10, 11, 12)


@dataclass
class ConfigEMG:
    n_trials: int = 29
    trials_to_plot: tuple = (1, 29)
    norm_trials = np.arange(6, 9).tolist()
    v_lines: tuple = (8.5, 26.5)
    pre_trials = np.arange(6, 9).tolist()
    train_trials = np.arange(9, 27).tolist()
    post_trials = np.arange(27, 30).tolist()
    output_folder: str = 'figures_emg_new'
    condition: tuple = ('Uncertainty', ['No', 'Tactile', 'Visual', 'Both'])
    name_length: tuple = (30, 12, 13, 14)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folder", type=str, default='./')
    parser.add_argument("-n", "--name", type=str, default="figure.png")
    parser.add_argument("-p", "--plot", type=str, default="point_plot")
    parser.add_argument('-norm', '--normalization', type=bool, default=False)
    parser.add_argument('-b', '--bits', type=bool, default=False)
    args = parser.parse_args()

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
        # breakpoint()
        if len(file_name) == config.name_length[0]:  #
            participant = int(
                file_name[config.name_length[1]:config.name_length[2]])
        else:
            participant = int(
                file_name[config.name_length[1]:config.name_length[3]])

        # Wrap condition
        condition = int(file_name[-5:-4])
        if config.__name__ == 'ConfigEMG':
            if condition % 2 == 0:
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

        else:
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
                                            config.norm_trials].mean() *
                                        100)
        tmp_df['NormTargetFeedforward'] = (tmp_df['TargetFeedforward'] /
                                           tmp_df['TargetFeedforward'].iloc[
                                               config.norm_trials].mean() * 100)
        tmp_df['NormTargetTotalInfo'] = (tmp_df['TargetTotalInfo'] /
                                         tmp_df['TargetTotalInfo'].iloc[
                                             config.norm_trials].mean() * 100)
        tmp_df['Condition'] = condition

        df = pd.concat((df, tmp_df))

    # breakpoint()
    colors = [
        (255, 182, 193),  # Cond 1
        (69, 249, 73),  # Cond 4
        (255, 0, 0),
        (5, 137, 8),
        (255, 0, 255),
        (0, 255, 127)
    ]

    colors_ = []
    [colors_.append('#%02x%02x%02x' % c) for c in colors]

    if args.plot == 'line_plot':
        line_plot(df, name=args.name, config=config,
                  normalization=args.normalization, colors=colors_,
                  bits=args.bits)
    elif args.plot == 'point_plot':
        point_plot(df, name=args.name, config=config,
                   normalization=args.normalization, colors=colors_,
                   bits=args.bits)
    return


def point_plot(df, config, name, normalization, colors, bits):
    if normalization:
        var = 'Norm'
    else:
        var = ''

    # df = df[(df['Stimulation'] == 'Sham')]
    # df = df[(df['Participant'] != 8) & (df['Participant'] != 22)]

    df_pre = df[(df['TrialNumber']).isin(config.pre_trials)]
    df_train = df[(df['TrialNumber']).isin(config.train_trials)]
    df_post = df[(df['TrialNumber']).isin(config.post_trials)]

    # Train
    df_pre = df_pre[[f'{var}TargetFeedback', f'{var}TargetFeedforward',
                     f'{var}TargetTotalInfo',
                     'Participant', 'Reinforcement', 'Condition',
                     f'{config.condition[0]}']].groupby(
        ['Reinforcement', 'Participant', f'{config.condition[0]}',
         'Condition']).mean()
    df_pre = df_pre.reset_index()
    if bits:
        df_pre[[f'{var}TargetFeedback', f'{var}TargetFeedforward',
                f'{var}TargetTotalInfo']] = df_pre[
                                                [f'{var}TargetFeedback',
                                                 f'{var}TargetFeedforward',
                                                 f'{var}TargetTotalInfo']] * 60

    df_train = df_train[[f'{var}TargetFeedback', f'{var}TargetFeedforward',
                         f'{var}TargetTotalInfo',
                         'Participant', 'Reinforcement', 'Condition',
                         f'{config.condition[0]}']].groupby(
        ['Reinforcement', 'Participant', f'{config.condition[0]}',
         'Condition']).mean()
    df_train = df_train.reset_index()
    if bits:
        df_train[[f'{var}TargetFeedback', f'{var}TargetFeedforward',
                  f'{var}TargetTotalInfo']] = df_train[
                                                  [f'{var}TargetFeedback',
                                                   f'{var}TargetFeedforward',
                                                   f'{var}TargetTotalInfo']] * 60

    df_post = df_post[[f'{var}TargetFeedback', f'{var}TargetFeedforward',
                       f'{var}TargetTotalInfo', 'Condition',
                       'Participant',
                       'Reinforcement', f'{config.condition[0]}']].groupby(
        ['Reinforcement', 'Participant', f'{config.condition[0]}',
         'Condition']).mean()
    df_post = df_post.reset_index()
    if bits:
        df_post[[f'{var}TargetFeedback', f'{var}TargetFeedforward',
                 f'{var}TargetTotalInfo']] = df_post[
                                                 [f'{var}TargetFeedback',
                                                  f'{var}TargetFeedforward',
                                                  f'{var}TargetTotalInfo']] * 60

    if config.__name__ == 'ConfigEMG':
        order = [1, 2, 3, 4, 5, 6, 7, 8]
        colors_ = []
        colors = [
            (255, 182, 193),  # Cond 1
            (69, 249, 73),  # Cond 4
            (255, 102, 102),
            (128, 255, 0),
            (255, 0, 255),
            (0, 255, 127),
            (204, 0, 0),
            (76, 153, 0)
        ]
        [colors_.append('#%02x%02x%02x' % c) for c in colors]
        colors = colors_
    else:
        order = [1, 4, 2, 5, 3, 6]
        # order = [3, 6]
        colors = colors
    with plt.style.context('seaborn-v0_8-paper'):
        sns.set_theme(style='whitegrid')
        sns.set_context('notebook')
        fig, ax = plt.subplots(2, 3, figsize=(12, 8),
                               sharex='col', sharey='row')
        sns.pointplot(df_pre,
                      y=f"{var}TargetFeedback", x='Condition',
                      ax=ax[0, 0], order=order,
                      errorbar='se', join=False, palette=colors)
        sns.pointplot(df_pre,
                      y=f"{var}TargetFeedforward", x='Condition',
                      ax=ax[1, 0], order=order,
                      errorbar='se', join=False, palette=colors)

        # Comment underneath to remove points
        # sns.swarmplot(data=df_train,
        #               y=f"{var}TargetFeedback",
        #               palette=['gray', 'gray'],
        #               x='Reinforcement',
        #               ax=ax[0, 0])
        # sns.lineplot(data=df_train,
        #              x="Reinforcement",
        #              y=f"{var}TargetFeedback",
        #              estimator=None,
        #              units="Participant",
        #              color='lightgray',
        #              ax=ax[0, 0])
        sns.pointplot(df_train,
                      y=f"{var}TargetFeedback", x='Condition',
                      ax=ax[0, 1], order=order,
                      errorbar='se', join=False, palette=colors)

        # sns.swarmplot(data=df_train[(df_train['Participant'] != 6)],
        #               y=f"{var}TargetFeedforward",
        #               palette=['gray', 'gray'],
        #               x='Reinforcement',
        #               ax=ax[1, 0])
        # sns.lineplot(data=df_train[(df_train['Participant'] != 6)],
        #              x="Reinforcement",
        #              y=f"{var}TargetFeedforward",
        #              estimator=None,
        #              units="Participant",
        #              color='lightgray',
        #              ax=ax[1, 0])
        sns.pointplot(df_train[(df_train['Participant'] != 6)],
                      y=f"{var}TargetFeedforward", x="Condition",
                      ax=ax[1, 1], order=order,
                      errorbar='se', join=False, palette=colors)

        ax[0, 0].set_title('Pre-Training')

        # sns.pointplot(df_post,
        #               y=f"{var}TargetFeedback", x="Condition",
        #               ax=ax[0, 1], order=order,
        #               errorbar='se', join=False, palette=colors)
        # sns.pointplot(df_post,
        #               y=f"{var}TargetFeedforward", x="Condition",
        #               ax=ax[1, 1], order=order,
        #               errorbar='se', join=False, palette=colors)
        # sns.pointplot(df_post,
        #               y=f"{var}TargetTotalInfo", x="Condition",
        #               ax=ax[2, 1], order=order,
        #               errorbar='se', join=False, palette=colors)
        ax[0, 1].set_title('Training')

        sns.pointplot(df_post,
                      y=f"{var}TargetFeedback", x='Condition',
                      ax=ax[0, 2], order=order,
                      errorbar='se', join=False, palette=colors)
        sns.pointplot(df_post,
                      y=f"{var}TargetFeedforward", x='Condition',
                      ax=ax[1, 2], order=order,
                      errorbar='se', join=False, palette=colors)

        ax[0, 2].set_title('Post-Training')

        for i in range(3):
            ax[1, i].set_xticklabels([
                f"ReinOFF-{config.condition[1][0]}",
                f"ReinON-{config.condition[1][0]}",
                f"ReinOFF-{config.condition[1][1]}",
                f"ReinON-{config.condition[1][1]}",
                f"ReinOFF-{config.condition[1][2]}",
                f"ReinON-{config.condition[1][2]}",
                f"ReinOFF-{config.condition[1][3]}",
                f"ReinON-{config.condition[1][3]}"
            ],
                rotation=45)
        plt.tight_layout()
        plt.savefig(f"./{config.output_folder}/boxplot/{name}")


def line_plot(df, config, name, normalization, colors, bits):
    if normalization:
        var = 'Norm'
    else:
        var = ''

    ## General plots
    # with plt.style.context('seaborn-v0_8-paper'):
    #     sns.set_theme(style='whitegrid', palette=['r', 'g'])
    #     sns.set_context('talk')
    #     fig, axes = plt.subplots(3, 1, figsize=(15, 10),
    #                              sharex='all')
    #     df_mean = df[[f'{var}TargetFeedback', f'{var}TargetFeedforward',
    #                   f'{var}TargetTotalInfo', 'Reinforcement',
    #                   'TrialNumber']].groupby([
    #         'TrialNumber', 'Reinforcement']).mean()
    #     df_mean = df_mean.reset_index()
    #
    #     df_sem = df[[f'{var}TargetFeedback', f'{var}TargetFeedforward',
    #                  f'{var}TargetTotalInfo', 'Reinforcement',
    #                  'TrialNumber']].groupby([
    #         'TrialNumber', 'Reinforcement']).sem()
    #     df_sem = df_sem.reset_index()
    #
    #     # breakpoint()
    #     for f, ax in zip(df_mean.columns[2:], axes):
    #         for rein in ['OFF', 'ON']:
    #             ax.plot(np.arange(1, config.n_trials + 1),
    #                     df_mean[df_mean['Reinforcement'] == rein][
    #                         f'{f}'],
    #                     label=rein)
    #             ax.fill_between(np.arange(1, config.n_trials + 1),
    #                             df_mean[df_mean['Reinforcement'] == rein][
    #                                 f'{f}'] -
    #                             df_sem[df_sem['Reinforcement'] == rein][
    #                                 f'{f}'],
    #                             df_mean[df_mean['Reinforcement'] == rein][
    #                                 f'{f}'] +
    #                             df_sem[df_sem['Reinforcement'] == rein][
    #                                 f'{f}'],
    #                             alpha=0.2
    #                             )
    #         ax.set_ylabel(f"{f}")
    #         [ax.axvline(x=pos, color='k', linestyle='--') for pos in
    #          config.v_lines]
    #         # breakpoint()
    #         r = matplotlib.patches.Rectangle((config.v_lines[0], 0),
    #                                          config.v_lines[1] -
    #                                          config.v_lines[0],
    #                                          5,
    #                                          color='gray',
    #                                          alpha=0.2)
    #         ax.add_patch(r)
    #     axes[-1].set_xlim([1, config.n_trials])
    #     axes[-1].legend()
    #
    #     plt.tight_layout()
    #     plt.savefig(f"./{config.output_folder}/line_plot/OFF_ON.png")

    # Uncertainty plots
    with plt.style.context('seaborn-v0_8-paper'):
        sns.set_theme(style='whitegrid', palette=['r', 'g'])
        sns.set_context('talk')
        fig, axes = plt.subplots(len(config.condition[1]), 2, figsize=(15, 10),
                                 sharex='all', sharey='col')

        df_mean = df[[f'{var}TargetFeedback', f'{var}TargetFeedforward',
                      f'{var}TargetTotalInfo', 'Reinforcement',
                      'TrialNumber', f'{config.condition[0]}']].groupby([
            'TrialNumber', 'Reinforcement', f'{config.condition[0]}']).mean()
        df_mean = df_mean.reset_index()

        df_sem = df[[f'{var}TargetFeedback', f'{var}TargetFeedforward',
                     f'{var}TargetTotalInfo', 'Reinforcement',
                     'TrialNumber', f'{config.condition[0]}']].groupby([
            'TrialNumber', 'Reinforcement', f'{config.condition[0]}']).sem()
        df_sem = df_sem.reset_index()

        if bits:
            df_mean[[f'{var}TargetFeedback', f'{var}TargetFeedforward',
                     f'{var}TargetTotalInfo']] = df_mean[
                                                     [f'{var}TargetFeedback',
                                                      f'{var}TargetFeedforward',
                                                      f'{var}TargetTotalInfo']] * 60

            df_sem[[f'{var}TargetFeedback', f'{var}TargetFeedforward',
                    f'{var}TargetTotalInfo']] = df_sem[
                                                    [f'{var}TargetFeedback',
                                                     f'{var}TargetFeedforward',
                                                     f'{var}TargetTotalInfo']] * 60

        if config.__name__ == 'ConfigEMG':
            order = [1, 2, 3, 4, 5, 6, 7, 8]
            colors_ = []
            colors = [(255, 182, 193),  # Cond 1
                      (69, 249, 73),  # Cond 4
                      (255, 102, 102),
                      (128, 255, 0),
                      (255, 0, 255),
                      (0, 255, 127),
                      (204, 0, 0),
                      (76, 153, 0)
                      ]
            [colors_.append('#%02x%02x%02x' % c) for c in colors]
            colors = colors_

        for f, idx in zip(df_mean.columns[3:5], [0, 1]):
            ax_ = axes[:, idx]
            color_counter = 0
            for ax, uncer in zip(ax_, config.condition[1]):
                for rein in ['OFF', 'ON']:
                    # breakpoint()
                    ax.plot(np.arange(1, config.n_trials + 1),
                            df_mean[(df_mean['Reinforcement'] == rein)
                                    & (df_mean[
                                           f'{config.condition[0]}'] == uncer)][
                                f'{f}'],
                            label=rein,
                            color=colors[color_counter])

                    ax.fill_between(np.arange(1, config.n_trials + 1),
                                    df_mean[(df_mean['Reinforcement'] == rein)
                                            & (df_mean[
                                                   f'{config.condition[0]}'] == uncer)][
                                        f'{f}'] -
                                    df_sem[(df_sem['Reinforcement'] == rein)
                                           & (df_sem[
                                                  f'{config.condition[0]}'] == uncer)][
                                        f'{f}'],
                                    df_mean[(df_mean['Reinforcement'] == rein)
                                            & (df_mean[
                                                   f'{config.condition[0]}'] == uncer)][
                                        f'{f}'] +
                                    df_sem[(df_sem['Reinforcement'] == rein)
                                           & (df_sem[
                                                  f'{config.condition[0]}'] == uncer)][
                                        f'{f}'],
                                    alpha=0.2,
                                    color=colors[color_counter]
                                    )
                    color_counter = color_counter + 1

                ax.set_ylabel(f"{uncer}")
                [ax.axvline(x=pos, color='k', linestyle='--') for pos in
                 config.v_lines]
                # r = matplotlib.patches.Rectangle((config.v_lines[0], 0),
                #                                  config.v_lines[1] -
                #                                  config.v_lines[0],
                #                                  5,
                #                                  color='gray',
                #                                  alpha=0.2)
                # ax.add_patch(r)
        axes[-1, -1].set_xlim([1, config.n_trials])
        axes[-1, -1].legend()
        axes[0, 0].set_title('TargetFeedback')
        axes[0, 1].set_title('TargetFeedforward')

        plt.tight_layout()
        plt.savefig(f"./{config.output_folder}/line_plot/{name}")

    return


if __name__ == '__main__':
    main()
