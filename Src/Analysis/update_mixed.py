import argparse
import glob
import os
import sys
import time

from itertools import product
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.nonparametric.api as smnp
import swifter

import utils
import graphs

N_PROC = 28

BASE_DIR = '/home/johnmcbride/projects/Scales/Data_compare/Processed/'
RAW_DIR  = '/home/johnmcbride/projects/Scales/Toy_model/Data/Raw/'
PRO_DIR  = '/home/johnmcbride/projects/Scales/Toy_model/Data/Processed/'
DIST_DIR  = '/home/johnmcbride/projects/Scales/Toy_model/Data/None_dist/'
REAL_DIR = os.path.join(BASE_DIR, 'Real')

PRO_DIR = "/media/johnmcbride/961391f1-186f-4345-881e-92d8bb3931c8/Projects/Scales/Results/Toy_model_scales/Processed/"
PRO_DIR2 = "/media/johnmcbride/961391f1-186f-4345-881e-92d8bb3931c8/Projects/Scales/Results/Toy_model_scales/Old_Processed/"


def process_grid_similar_scales(df, df_real, n):
    timeS = time.time()
    idx = df_real.loc[df_real.n_notes==n].index
    for w in [10, 20]:
        df[f'mss_w{w:02d}'] = df.mix_scale.apply(lambda x: ss_fn(x, df_real, idx, w))
        print(f"ss_w{w:02d}: {(time.time()-timeS)/60.} minutes")
    return df


def ss_fn(x, df_real, idx, w):
    return ';'.join([str(i) for i in idx if is_scale_similar(x, df_real.loc[i, 'scale'], w)])


def is_scale_similar(x, y, w):
    xint = [int(a) for a in x.split(';')]
    yint = [int(a) for a in y.split(';')]
    return np.allclose(xint, yint, atol=w)


def how_much_real_scales_predicted(df, n_real, w):
    return float(len(set([int(x) for y in df[f"ss_w{w:02d}"] for x in y.split(';') if len(y)]))) / float(n_real)



if __name__ == "__main__":

    timeS = time.time()

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--partabase', action='store', default='None', type=str)
    args = parser.parse_args()

    categories = ['pair_ints', 'scale']
    n_arr = np.arange(4,10,dtype=int)

    df_model = pd.read_feather(os.path.join(BASE_DIR, 'monte_carlo_comparison.feather'))
    df_real = pd.read_pickle(os.path.join(REAL_DIR, 'real_scales.pickle'))

    files = df_model.loc[(df_model.mfm_10.notnull()), 'fName']

    print(f"Real scales loaded after {(time.time()-timeS)/60.} minutes")


    def read_model_results(path):
        try:
            fName = os.path.split(path)[1]
            n = int(fName.split('_')[0].strip('n'))
            df = pd.read_feather(path)
            df = process_grid_similar_scales(df, df_real, n)
            df.to_feather(path)
        except Exception as e:
            print(f"{path}\n{e}")


    print(f"MC scales loaded after {(time.time()-timeS)/60.} minutes")


    pool = mp.Pool(N_PROC)

    results = list(pool.imap_unordered(read_model_results, files))
    print(f"Model comparison finished after {(time.time()-timeS)/60.} minutes")



