from itertools import product
import os

import matplotlib.pyplot as plt
from multiprocessing import Pool
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu

import process_csv
from process_csv import DATA_DIR
import utils

N_PROC = 60


def load_text_summary():
    df = pd.read_excel('../scales_database.xlsx', "source_list")
    return df["Supports octave equivalence"]


def instrument_tunings():
    # Data from the ... book with measured instrument intervals
    df_2 = pd.read_csv(os.path.join(DATA_DIR,'scales_B.csv'))
    # Data from papers with measured instrument intervals
    df_3 = pd.read_csv(os.path.join(DATA_DIR,'scales_C.csv'))
    df_3 = utils.reformat_original_csv_data(df_3)
    # Data from Ellis (1885)
    df_5 = pd.read_csv(os.path.join(DATA_DIR,'scales_E.csv'))
    df_5 = utils.reformat_original_csv_data(df_5)

    df = pd.concat([df_2, df_3, df_5]).reset_index(drop=True)
    df['scale'] = df.Intervals.apply(lambda x: np.cumsum(utils.str_to_ints(x)))
    df['max_scale'] = df.scale.apply(max)
    return df


def octave_chance(df, n_rep=10, plot=False, octave=1200, w=50):
    df = df.loc[df.scale.apply(lambda x: x[-2] >= octave-w)]
    print(len(df))

    ints = df.Intervals.apply(utils.str_to_ints).values
#   all_ints = np.array([x for y in ints for x in np.cumsum(y)])
    all_ints = np.array([x for y in ints for i in range(len(y)) for x in np.cumsum(y[i:])])
    oct_real = all_ints[(all_ints>=octave-w)&(all_ints<=octave+w)]
    print(len(oct_real), len(oct_real) / len(all_ints))

    shuffled_ints = []
    for j in range(n_rep):
        for i in ints:
            ran = np.random.choice(i, replace=False, size=len(i))
#           for k in np.cumsum(ran):
#               shuffled_ints.append(k)
            for k in range(len(ran)):
                for m in np.cumsum(ran[k:]):
                    shuffled_ints.append(m)

    shuffled_ints = np.array(shuffled_ints)
    idx = (shuffled_ints>=octave-w)&(shuffled_ints<=octave+w)
    oct_shuf = shuffled_ints[idx]
    print(len(oct_shuf) / len(shuffled_ints))
    
    if plot:
        fig, ax = plt.subplots(1,2)
        sns.distplot(np.abs(oct_real-octave), bins=np.arange(0, w+10, 10), kde=False, norm_hist=True, ax=ax[0])
        sns.distplot(np.abs(oct_shuf-octave), bins=np.arange(0, w+10, 10), kde=False, norm_hist=True, ax=ax[0])
        sns.distplot(oct_real, bins=np.arange(octave-w, octave+w+10, 10), kde=False, norm_hist=True, ax=ax[1])
        sns.distplot(oct_shuf, bins=np.arange(octave-w, octave+w+10, 10), kde=False, norm_hist=True, ax=ax[1])

    print(mannwhitneyu(np.abs(oct_real-octave), np.abs(oct_shuf-octave)))
    print(np.mean(np.abs(oct_real-octave)))
    print(np.mean(np.abs(oct_shuf-octave)))


def label_sig(p):
    if p >= 0.05:
        return "x"
    elif p >= 0.005:
        return '*'
    elif p >= 0.0005:
        return '**'
    elif p >= 0.00005:
        return '***'


def octave_chance_individual(df, n_rep=10, plot=False, octave=1200, w1=100, w2=20):
    df = df.loc[df.scale.apply(lambda x: x[-2] >= octave)]
    ints = df.Intervals.apply(utils.str_to_ints).values

    res = pd.DataFrame(columns=["max_scale", "n_notes", "ints", "oct_real", "oct_shuf", "mean_real", "mean_shuf", "MWU", "f_real", "f_shuf"])

    for i in ints:
        all_ints = np.array([x for j in range(len(i)) for x in np.cumsum(i[j:])])
        oct_real = all_ints[(all_ints>=octave-w1)&(all_ints<=octave+w1)]
        f_real = sum(np.abs(all_ints-octave)<=w2) / len(all_ints)
        mean_real = np.mean(np.abs(oct_real-octave))

        shuffled_ints = []
        for j in range(n_rep):
            ran = np.random.choice(i, replace=False, size=len(i))
            for k in range(len(ran)):
                for m in np.cumsum(ran[k:]):
                    shuffled_ints.append(m)
        shuffled_ints = np.array(shuffled_ints)
        idx = (shuffled_ints>=octave-w1)&(shuffled_ints<=octave+w1)
        oct_shuf = shuffled_ints[idx]
        f_shuf = sum(np.abs(shuffled_ints-octave)<=w2) / len(shuffled_ints)
        mean_shuf = np.mean(np.abs(oct_shuf-octave))

        try:
            mwu = mannwhitneyu(np.abs(oct_real-octave), np.abs(oct_shuf-octave))[1]
        except ValueError:
            mwu = 1
        res.loc[len(res)] = [sum(i), len(i), i, oct_real, oct_shuf, mean_real, mean_shuf, mwu, f_real, f_shuf]

    res['sig'] = res.MWU.apply(label_sig)
    return res


def create_new_scales(df, n_rep=10):
    ints = [x for y in df.Intervals.apply(utils.str_to_ints) for x in y]
    n_notes = df.scale.apply(len).values
    df_list = []

    for i in range(n_rep):
        new_ints = [utils.ints_to_str(np.random.choice(ints, replace=True, size=n)) for n in n_notes]
        new_df = df.copy()
        new_df.Intervals = new_ints
        df_list.append(new_df)

    return df_list


def get_stats(df, i, k, w1=100, w2=20, n_rep=50, nrep2=100):
    out = np.zeros((3,nrep2), float)
    for j in range(nrep2):
        res = octave_chance_individual(df, octave=i, n_rep=n_rep, w1=w1, w2=w2)
        out[0,j] = len(res.loc[(res.MWU<0.05)&(res.mean_real<res.mean_shuf)])
        out[1,j] = len(res.loc[(res.MWU<0.05)&(res.mean_real>res.mean_shuf)])
        out[2,j] = len(res.loc[(res.MWU>=0.05)])
    np.save(f"../IntStats/{k}_w1{w1}_w2{w2}_I{i:04d}.npy", out)
    return out.mean(axis=1)


def unexpected_intervals(df):
    df = df.loc[:, ['Intervals', 'scale']]
    ints = np.arange(200, 2605, 5)
    w1_list = [50, 75, 100, 125, 150, 175, 200]
    w2_list = [5, 10, 15, 20, 30, 40]
    for w1 in w1_list:
        for w2 in w2_list:
            with Pool(N_PROC) as pool:
                res = pool.starmap(get_stats, product([df], ints, [0], [w1], [w2]), 9)

    for c in df.Continent.unique():
        alt_df = df.loc[df.Continent!=c]
        with Pool(N_PROC) as pool:
            res = pool.starmap(get_stats, product([alt_df], ints, [c], [w1], [w2]), 9)
        

#   alt_df = create_new_scales(df, n_rep=3)
#   with Pool(N_PROC) as pool:
#       for i in range(10):
#           res = pool.starmap(get_stats, product([alt_df[i]], ints, [i+1]), 9)


if __name__ == "__main__":

    df = instrument_tunings()
    unexpected_intervals(df)


