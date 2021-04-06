from itertools import product
import os

import matplotlib.pyplot as plt
from multiprocessing import Pool
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu, lognorm, norm

import process_csv
from process_csv import DATA_DIR
import utils

N_PROC = 60


def load_text_summary():
    df = pd.read_excel('../scales_database.xlsx', "source_list")
    Y1 = "Players exhibit octave?"
    Y2 = "Sources indicate that octave is generally used in culture?"
    for Y in [Y1, Y2]:
        df.loc[df[Y].isnull(), Y] = ''
    return df.loc[:, [Y1, Y2]]


def get_md2(ints):
    if isinstance(ints, str):
        ints = np.array([float(x) for x in ints.split(';')])
    return np.min([np.sum(np.roll(ints, i)[:2]) for i in range(len(ints))])
#   md2 = np.array([np.sum(np.roll(poss, i, axis=1)[:,:2], axis=1) for i in range(7)]).min(axis=0)




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
    df['Intervals'] = df.Intervals.apply(lambda x: utils.str_to_ints(x))
    df['max_scale'] = df.scale.apply(max)
    df['min_int'] = df.Intervals.apply(min)
    df['max_int'] = df.Intervals.apply(max)
    df['AllInts'] = df.Intervals.apply(lambda x: [y for i in range(len(x)-1) for y in np.cumsum(x[i:])])
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
    ints = df.Intervals.values

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


def get_norm_posterior(Y, s, m):
    n = len(Y)
    sy = np.sum(Y)
    sy2 = np.sum(np.square(Y))
    a = n / (2 * s**2)
    b = sy / (s**2)
    c = - sy2 / (2 * s**2)
    A = 0.5 * (sy2 + n * m**2 - 2 * m * sy)
    left = (a/np.pi)**0.5 * np.exp(-a * m**2 + b * m - b**2 / (4*a))
    right = A**(n/2) / (2*np.pi*n) * np.exp(-A / s**2 - n*np.log(n)-1) / s**(n+2)
    return left * right


def evaluate_best_fit_lognorm(df):
    Y = [x for c in df.Continent.unique() for y in np.random.choice(df.loc[df.Continent==c, "AllInts"], size=6) for x in y]
    Yl = np.log(np.array(Y))
    s_arr = np.linspace(0, 2, 1001)[1:]
    m_arr = np.linspace(np.log(25), np.log(6000), 1001)
    si, mi = np.meshgrid(s_arr, m_arr)
    return get_norm_posterior(Yl, si, mi)


def get_int_prob_via_sampling(df, ysamp='AllInts', xsamp='Continent', s=6, ax=''):
    if len(xsamp):
        Y = [x for c in df[xsamp].unique() for y in np.random.choice(df.loc[df[xsamp]==c, ysamp], size=s) for x in y]
    else:
        Y = [x for y in df[ysamp] for x in y]
#   Yl = np.log(np.array(Y))
#   print(norm.fit(Yl))

    bins = np.arange(0, 5000, 20)
    X = bins[:-1] + np.diff(bins[:2])/2

#   shape, loc, scale = lognorm.fit(Y)
    shape, loc, scale = [0.93, -45.9, 605.4]
    params = lognorm.fit(Y, loc=loc, scale=scale)
    print(params)
    boot = np.array([np.histogram(lognorm.rvs(*params, len(Y)), bins=bins, density=True)[0] for i in range(10000)])

    if isinstance(ax, str):
        fig, ax = plt.subplots()
    ax.plot(X, np.histogram(Y, bins=bins, density=True)[0], '-', c=sns.color_palette()[0])
    ax.plot(X, lognorm.pdf(X, *params), ':k')
    ax.fill_between(X, *[np.quantile(boot, q, axis=0) for q in [0.01, 0.99]], color='pink')

    


if __name__ == "__main__":

    df = instrument_tunings()
    unexpected_intervals(df)


