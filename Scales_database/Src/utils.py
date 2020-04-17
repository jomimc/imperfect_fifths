
import re
import sys
import time

import matplotlib.pyplot as plt
from itertools import permutations
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import DBSCAN
import statsmodels.nonparametric.api as smnp
import swifter


#############################################################################
### Parameters


### Theoretical scale markers
### PYT = Pythagorean tuning
### EQ5 = 5-Tone Equal Temperament
### JI  = Just intonation
### CHINA = Shi-er-lu
### The rest are sourced from Rechberger, Herman

PYT_INTS = np.array([0., 90.2, 203.9, 294.1, 407.8, 498.1, 611.7, 702., 792.2, 905., 996.1, 1109.8, 1200.])
EQ5_INTS = np.linspace(0, 1200, num=6, endpoint=True, dtype=float)
EQ7_INTS = np.linspace(0, 1200, num=8, endpoint=True, dtype=float)
EQ9_INTS = np.linspace(0, 1200, num=10, endpoint=True, dtype=float)
EQ10_INTS = np.linspace(0, 1200, num=11, endpoint=True, dtype=float)
EQ12_INTS = np.linspace(0, 1200, num=13, endpoint=True, dtype=float)
EQ24_INTS = np.linspace(0, 1200, num=25, endpoint=True, dtype=float)
EQ53_INTS = np.linspace(0, 1200, num=54, endpoint=True, dtype=float)
JI_INTS = np.array([0., 111.7, 203.9, 315.6, 386.3, 498.1, 590.2, 702., 813.7, 884.4, 1017.6, 1088.3, 1200.])
SLENDRO = np.array([263., 223., 253., 236., 225.])
PELOG   = np.array([167., 245., 125., 146., 252., 165., 100.])
DASTGAH = np.array([0., 90., 133.23, 204., 294.14, 337.14, 407.82, 498., 568.72, 631.28, 702., 792.18, 835.2, 906., 996., 1039.1, 1109.77, 1200.])
TURKISH = {'T':203.8, 'K':181.1, 'S':113.2, 'B':90.6, 'F':22.6, 'A':271, 'E':67.9}
KHMER_1 = np.array([185., 195., 105., 195., 195., 185., 140.])
KHMER_2 = np.array([190., 190., 130., 190., 190., 190., 120.])
VIET    = np.array([0., 175., 200., 300., 338., 375., 500., 520., 700., 869., 900., 1000., 1020., 1200.])
CHINA   = np.array([0., 113.67291609,  203.91000173,  317.73848174,  407.83554758, 520.68758457,  611.71791523,  701.95500087,  815.62791696, 905.8650026 , 1019.47514332, 1109.76982292, 1201.27828039])


### Maximum allowable deviation from a perfect octave
### i.e., scale is included if the intervals sum to 1200 +- OCT_CUT
OCT_CUT = 50


#############################################################################
### Functions to be used in reformatting the data

def get_cents_from_ratio(ratio):
    return 1200.*np.log10(ratio)/np.log10(2)




#############################################################################
### Functions for extracting and reformatting the data


### Encode a scale as a binary string:
### If the first character is 0, then the first potential note in the scale is
### not played. If it is 1, then it is played.
### E.g. The major scale in 12-TET is given by 010110101011
### The intervals are then retrieved by comparing the mask with the correct tuning system
def reformat_scales_as_mask(df):
    st = '000000000000001'
    fn = lambda x: '1' + ''.join([st[-int(i):] for i in x])
    idx = df.loc[df.Tuning.apply(lambda x: x not in ['Unique', 'Turkish', '53-tet'])].index
    df.loc[idx, 'mask'] = df.loc[idx, 'Intervals'].apply(fn)

    fn = lambda x: '1' + ''.join([st[-int(i):] for i in x.split(';')])
    idx = df.loc[df.Tuning=='53-tet'].index
    df.loc[idx, 'mask'] = df.loc[idx, 'Intervals'].apply(fn)
    return df


def reformat_surjodiningrat(df):
    for row in df.itertuples():
        ints = [get_cents_from_ratio(float(row[i+3])/float(row[i+2])) for i in range(7) if row[i+3] != 0]
        df.loc[row[0], 'pair_ints'] = ';'.join([str(int(round(x))) for x in ints])
    df['Reference'] = 'Surjodiningrat'
    df['Theory'] = 'N'
    df = df.drop(columns=[str(x) for x in range(1,9)])
    return df


def reformat_original_csv_data(df):
    new_df = pd.DataFrame(columns=['Name', 'Intervals', 'Culture', 'Continent', 'Tuning', 'Reference', 'Theory'])
    for i, col in enumerate(df.columns):
        tuning  = df.loc[0, col]
        culture = df.loc[1, col]
        cont    = df.loc[2, col]
        ref     = df.loc[3, col]
        theory  = df.loc[4, col]
        try:
            int(col)
            name = '_'.join([culture, col])
        except:
            name = col
        ints = ';'.join([str(int(round(float(x)))) for x in df.loc[5:, col] if not str(x)=='nan'])
        new_df.loc[i] = [name, ints, culture, cont, tuning, ref, theory]
    return new_df


def extract_scales_and_ints_from_scales(df):
    names = []
    scales = []
    all_ints = []
    pair_ints = []
    cultures = []
    tunings = []
    conts = []
    ref = []
    theory = []
    for row in df.itertuples():
        try:
            idx = np.where(np.array([int(x) for x in row.mask]))[0]
        except:
            pass

        for tun in row.Tuning.split(';'):

            if tun == '12-tet':
                scale = EQ12_INTS[idx]
            elif tun == '53-tet':
                scale = EQ53_INTS[idx]
            elif tun == 'Just':
                scale = JI_INTS[idx]
            elif tun == 'Pythagorean':
                scale = PYT_INTS[idx]
            elif tun == 'Arabian':
                scale = EQ24_INTS[idx]
            elif tun == 'Dastgah-ha':
                scale = DASTGAH[idx]
            elif tun == 'Vietnamese':
                scale = VIET[idx]
            elif tun == 'Chinese':
                scale = CHINA[idx]
            elif tun == 'Turkish':
                scale = np.cumsum([0.0] + [TURKISH[a] for a in row.Intervals])
            elif tun == 'Khmer':
                for KHM in [KHMER_1, KHMER_2]:
                    base = KHM[[i-1 for i in idx[1:]]]
                    for i in range(len(base)):
                        scale = np.cumsum([0.] + np.roll(KHM,i))
                        names.append(row.Name)
                        scales.append(scale)
                        all_ints.append([scale[i] - scale[j] for j in range(len(scale)) for i in range(j+1,len(scale))])
                        pair_ints.append([scale[j+1] - scale[j] for j in range(len(scale)-1)])
                        cultures.append(row.Culture)
                        tunings.append(tun)
                        conts.append(row.Continent)
                        ref.append(row.Reference)
                        theory.append(row.Theory)
                continue
            elif tun == 'Unique':
                scale = np.cumsum([0.] + [float(x) for x in row.Intervals.split(';')])
            else:
                print(row.Name, tun, tun=='12-tet')
                continue

            names.append(row.Name)
            scales.append(scale)
            all_ints.append([scale[i] - scale[j] for j in range(len(scale)) for i in range(j+1,len(scale))])
            pair_ints.append([scale[j+1] - scale[j] for j in range(len(scale)-1)])
            cultures.append(row.Culture)
            tunings.append(tun)
            conts.append(row.Continent)
            ref.append(row.Reference)
            theory.append(row.Theory)

    return cultures, tunings, conts, names, scales, all_ints, pair_ints, ref, theory


def extract_scales_and_ints_from_unique(df):
    names = []
    scales = []
    all_ints = []
    pair_ints = []
    cultures = []
    tunings = []
    conts = []
    ref = []
    theory = []
    for row in df.itertuples():
        ints = [int(x) for x in row.Intervals.split(';')]
        if sum(ints) < (1200 - OCT_CUT):
            continue

        start_from = 0
        for i in range(len(ints)):
            if i < start_from:
                continue
            sum_ints = np.cumsum(ints[i:], dtype=int)
            # If the total sum of ints is less than the cutoff, ignore this entry
            if sum_ints[-1] < (1200 - OCT_CUT):
                break
            # Find the scale degree by finding the note closest to 1200
            idx_oct = np.argmin(np.abs(sum_ints-1200))
            oct_val = sum_ints[idx_oct]
            # If the total sum of ints is greater than the cutoff, move
            # on to the next potential scale
            if abs(oct_val - 1200) > OCT_CUT:
                continue
            
            scale = [0.] + list(sum_ints[:idx_oct+1])
            names.append(row.Name)
            scales.append(scale)
            all_ints.append([scale[i] - scale[j] for j in range(len(scale)) for i in range(j+1,len(scale))])
            pair_ints.append([scale[j+1] - scale[j] for j in range(len(scale)-1)])
            cultures.append(row.Culture)
            tunings.append(row.Tuning)
            conts.append(row.Continent)
            ref.append(row.Reference)
            theory.append('N')

            # When searching for new scales from this entry, start from
            # this index
            start_from = idx_oct + i

    return cultures, tunings, conts, names, scales, all_ints, pair_ints, ref, theory


