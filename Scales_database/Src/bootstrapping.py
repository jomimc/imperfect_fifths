import os
from pathlib import Path
import pickle

from multiprocessing import Pool
import numpy as np
import pandas as pd

import utils


PATH_DATA = Path("../Figures/Data")


def boot_hist(df, xsamp, ysamp, bins, s=0, n_rep=1000):
    Y = []
    for i in range(n_rep):
        if len(xsamp):
            y = []
            for c in df[xsamp].unique():
                all_y = df.loc[df[xsamp]==c, ysamp].values
                y.extend(list(np.random.choice(all_y, replace=True, size=min(s, len(all_y)))))
        else:
            y = np.random.choice(df[ysamp].values, replace=True, size=len(df))
        Y.append(np.histogram(y, bins=bins, density=True)[0])
    Y = np.array(Y)
    return Y.mean(axis=0), np.quantile(Y, 0.025, axis=0), np.quantile(Y, 0.975, axis=0)


def scale_degree(df, n_rep=1000):
    bins = np.arange(3.5, 10, 1)
    X = np.arange(4, 10, 1)
    out = {'X': X,
           'All': boot_hist(df, '', 'n_notes', bins),
           'Theory': boot_hist(df.loc[df.Theory=='Y'], '', 'n_notes', bins),
           'Measured': boot_hist(df.loc[df.Theory=='N'], '', 'n_notes', bins)}

    xsamp_list = ['Continent', 'Culture']
    for xsamp, s in zip(xsamp_list, [10, 5]):
        out.update({xsamp: boot_hist(df, xsamp, 'n_notes', bins, s=s)})

    pickle.dump(out, open(PATH_DATA.joinpath("scale_degree.pickle"), 'wb'))
    return out


def unfold_list(l):
    return [x for y in l for x in y]


def boot_hist_list(df, xsamp, ysamp, bins, s=0, n_rep=1000):
    Y = []
    for i in range(n_rep):
        if len(xsamp):
            y = []
            for c in df[xsamp].unique():
                all_y = df.loc[df[xsamp]==c, ysamp].values
                y.extend(unfold_list(np.random.choice(all_y, replace=True, size=min(s, len(all_y)))))
        else:
            y = unfold_list(np.random.choice(df[ysamp].values, replace=True, size=len(df)))
        Y.append(np.histogram(y, bins=bins, density=True)[0])
    Y = np.array(Y)
    return Y.mean(axis=0), np.quantile(Y, 0.025, axis=0), np.quantile(Y, 0.975, axis=0)


def boot_list(df, ysamp='pair_ints'):
    if isinstance(df.loc[0, ysamp], str):
        df[ysamp] = df[ysamp].apply(utils.str_to_ints)

    bins = {'pair_ints':np.arange(-10, 520, 20),
            'scale': np.arange(15, 1270, 30),
            'all_ints2': np.arange(15, 1270, 30)}[ysamp]

    X = bins[1:] - np.diff(bins[:2]) * 0.5
    out = {'X': X,
           'All': boot_hist_list(df, '', ysamp, bins),
           'Theory': boot_hist_list(df.loc[df.Theory=='Y'], '', ysamp, bins),
           'Measured': boot_hist_list(df.loc[df.Theory=='N'], '', ysamp, bins)}
    
    xsamp_list = ['Continent', 'Culture']
    for xsamp, s in zip(xsamp_list, [10, 5]):
        out.update({xsamp: boot_hist_list(df, xsamp, ysamp, bins, s=s)})

    stem = {'pair_ints':'adjacent_int',
            'scale':'scale',
            'all_ints2':'all_int'}[ysamp]
    pickle.dump(out, open(PATH_DATA.joinpath(f"{stem}.pickle"), 'wb'))
    return out


