import os

import matplotlib as mpl 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import utils

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

DATA_DIR = '/home/johnmcbride/projects/Scales/Book1/Data/'
FIGS_DIR = '/home/johnmcbride/Dropbox/phd/LaTEX/Scales/Figures/'

EQ12_INTS = np.linspace(0, 1200, num=13, endpoint=True, dtype=float)
JI_INTS = np.array([0., 111.7, 203.9, 315.6, 386.3, 498.1, 590.2, 702., 813.7, 884.4, 1017.6, 1088.3, 1200.])

### Create new DataFrame with intervals in cents for all tunings

def reformat_scales_as_mask(df):
    st = '000000000000001'
    fn = lambda x: '1' + ''.join([st[-int(i):] for i in x])
    idx = df.loc[df.Tuning.apply(lambda x: x not in ['Unique', 'Turkish', '53-tet'])].index
    df.loc[idx, 'mask'] = df.loc[idx, 'Intervals'].apply(fn)

    fn = lambda x: '1' + ''.join([st[-int(i):] for i in x.split(';')])
    idx = df.loc[df.Tuning=='53-tet'].index
    df.loc[idx, 'mask'] = df.loc[idx, 'Intervals'].apply(fn)
    return df

def extract_scales_and_ints_from_scales(df):
    names = []
    all_ints = []
    pair_ints = []
    n_notes = []
    tunings = []
    for row in df.itertuples():
        try:
            idx = np.where(np.array([int(x) for x in row.mask]))[0]
        except:
            pass

        for tun in row.Tuning.split(';'):

            if tun == '12-tet':
                for scale, tun in zip([EQ12_INTS[idx], JI_INTS[idx]], ['EQ', 'JI']):
                    names.append(row.Name)
#                   scales.append(scale)
                    pair_ints.append([scale[j+1] - scale[j] for j in range(len(scale)-1)])
                    all_ints.append([x for i in range(len(pair_ints[-1])) for x in np.cumsum(np.roll(pair_ints[-1],i))[:-1]])
#                   cultures.append(row.Culture)
                    tunings.append(tun)
                    n_notes.append(len(scale)-1)
#                   conts.append(row.Continent)
#                   ref.append(row.Reference)
#                   theory.append(row.Theory)
    str_ints = [';'.join([str(int(round(x))) for x in y]) for y in pair_ints]
    str_all_ints = [';'.join([str(int(round(x))) for x in y]) for y in all_ints]

    return pd.DataFrame(data={'Name':names, 'n_notes':n_notes, 'pair_ints':str_ints, 'all_ints':str_all_ints, 'tuning':tunings})

#   return cultures, tunings, conts, names, scales, all_ints, pair_ints, ref, theory

def get_harmonic_score(all_ints, w):
    ints = [int(x) for x in all_ints.split(';')]
    return np.mean([get_similarity_of_nearest_attractor(x, ATTRACTORS[w][1], ATTRACTORS[w][3]) for x in ints])

def get_similarity_of_nearest_attractor(x, sc_f, simil):
    minIdx = np.argmin(np.abs(sc_f - x))
    return simil[minIdx]

def calculate_most_harmonic_neighbour(int_cents, sim_only=False, CENT_DIFF_MAX=22):
    best_ratio = [1,1]
    max_similarity = 0.0
    cents = 0.0
    for x in np.arange(1,75, dtype=float):
        cent_diff = 1200.*np.log10((x+1.)/x)/np.log10(2.) - int_cents
        if cent_diff > CENT_DIFF_MAX:
            continue
        for y in np.arange(x+1.,99., dtype=float):
            cent_diff = abs(1200.*np.log10(y/x)/np.log10(2.)- int_cents)
            if cent_diff > CENT_DIFF_MAX:
                continue
            simil = ((x+y-1.)/(x*y))*100.
            if simil > max_similarity:
                cents = 1200.*np.log10(y/x)/np.log10(2.)
                best_ratio = [y,x]
                max_similarity = simil
    if sim_only:
        return max_similarity
    else:
        return max_similarity, best_ratio, cents

def get_attractors(n, dI=5., diff=22):
    sc_i = np.arange(dI, 1200.+dI, dI) 
    sc_f = set()
    attract = []
    ratios = []
    simils = []
    for s in sc_i:
        max_similarity, best_ratio, cents = calculate_most_harmonic_neighbour(s, CENT_DIFF_MAX=diff)
        if max_similarity == 0.0:
            continue
        if round(cents,2) not in sc_f:
            sc_f.add(round(cents,2))
            attract.append(round(cents,2))
            ratios.append(best_ratio)
            simils.append(max_similarity ** n / 100.**(n-1))
    return sc_i, np.array(attract), ratios, simils

ATTRACTORS = {w:get_attractors(n, diff=w) for n in [1] for w in [5,10,15,20]}

def plot_EQ_vs_JI():
    if os.path.exists(os.path.join(DATA_DIR, 'eq_vs_ji.feather')):
        df = pd.read_feather(os.path.join(DATA_DIR, 'eq_vs_ji.feather'))
    else:
        df = pd.read_csv(os.path.join(DATA_DIR,'scales_A.csv'))
        df = reformat_scales_as_mask(df)

        df = extract_scales_and_ints_from_scales(df)
        for w in range(5,25,5): df[f"w{w:02d}"] = df.all_ints.apply(lambda x: get_harmonic_score(x, w))
        df.to_feather(os.path.join(DATA_DIR, 'eq_vs_ji.feather'))

    fig, ax = plt.subplots(1,4, figsize=(10,5))

    for i, w in enumerate(range(5,25,5)):
        sns.distplot(df.loc[df.tuning=='EQ', f"w{w:02d}"], label='EQ', ax=ax[i])
        sns.distplot(df.loc[df.tuning=='JI', f"w{w:02d}"], label='JI', ax=ax[i])

        ax[i].set_xlabel(r'$HSS(w={0})$'.format(w*2))
        ax[i].set_yticks([])

    ax[0].set_ylabel("Normalised probability distribution")
    ax[0].legend(loc='best', frameon=False)
    plt.savefig(FIGS_DIR + 'eq_vs_ji.pdf', bbox_inches='tight', pad_inches=0)

if __name__ == '__main__':

    if 0:
        df = pd.read_csv(os.path.join(DATA_DIR,'scales_A.csv'))
        df = utils.reformat_scales_as_mask(df)

        df = extract_scales_and_ints_from_scales(df)
        for w in range(5,25,5): df[f"w{w:02d}"] = df.all_ints.apply(lambda x: get_harmonic_score(x, w))

        fig, ax = plt.subplots(4,1)

        for i, w in enumerate(range(5,25,5)):
            sns.distplot([y for x in df.loc[df.tuning=='EQ', f"w{w:02d}"] for y in utils.extract_ints_from_string(x)], label='EQ', ax=ax[i])
            sns.distplot([y for x in df.loc[df.tuning=='JI', f"w{w:02d}"] for y in utils.extract_ints_from_string(x)], label='JI', ax=ax[i])


