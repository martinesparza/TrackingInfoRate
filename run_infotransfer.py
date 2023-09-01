"""
Author: @mesparza
Run IT pipeline
"""
import argparse
import glob
import os

import pandas as pd

from ITscript import compute_info_transfer

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("-a", "--average", type=str, default='cond')
parser.add_argument("-p", "--path", type=str, default='data/')
args = parser.parse_args()


def main():

    # Files
    df = pd.DataFrame()
    files = os.listdir(f"{os.getcwd()}/preprocessed_data/")

    for file in files:
        df = pd.concat((df, compute_info_transfer(args.path, filename)))
    return


if __name__ == '__main__':
    main()