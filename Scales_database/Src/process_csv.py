import os

import numpy as np
import pandas as pd

import utils

DATA_DIR = '/home/johnmcbride/projects/Scales/temp/Scales_database/Data/'
#DATA_DIR = 'Data/'

### Load the data from csv files
def load_data():
    # Data from two books which cover mostly scales
    # which have fixed theory or scales
    df_1 = pd.read_csv(os.path.join(DATA_DIR,'scales_A.csv'))
    df_1 = utils.reformat_scales_as_mask(df_1)

    # Data from the ... book with measured instrument intervals
    df_2 = pd.read_csv(os.path.join(DATA_DIR,'scales_B.csv'))

    # Data from papers with measured instrument intervals
    df_3 = pd.read_csv(os.path.join(DATA_DIR,'scales_C.csv'))
    df_3 = utils.reformat_original_csv_data(df_3)

    # Data from Surjodiningrat et al. (1972)
    df_4 = pd.read_csv(os.path.join(DATA_DIR,'scales_D.csv'))
    df_4 = utils.reformat_surjodiningrat(df_4)

    return df_1, df_2, df_3, df_4


### Create new DataFrame with intervals in cents for all tunings

def reformat_data(old_dfs):
    new_dfs = []

    for i, df in enumerate(old_dfs):
        if i==0:
            cultures, tunings, conts, names, scales, all_ints, pair_ints, ref, theory = utils.extract_scales_and_ints_from_scales(df)
            str_ints = [';'.join([str(int(round(x))) for x in y]) for y in pair_ints]
        elif i==3:
            cultures, tunings, conts, names, str_ints, ref, theory = [df.loc[:,key] for key in ['Culture', 'Tuning', 'Continent', 'Name', 'pair_ints','Reference', 'Theory']]
            pair_ints = [[int(x) for x in y.split(';')] for y in str_ints]
            scales = [[0] + list(np.cumsum(ints)) for ints in pair_ints]
            all_ints = [[s[i] - s[j] for j in range(len(s)) for i in range(j+1,len(s))] for s in scales]
        else:
            cultures, tunings, conts, names, scales, all_ints, pair_ints, ref, theory = utils.extract_scales_and_ints_from_unique(df)
            str_ints = [';'.join([str(int(round(x))) for x in y]) for y in pair_ints]

        str_all_ints = [';'.join([str(int(round(x))) for x in y]) for y in all_ints]
        str_scales = [';'.join([str(int(round(x))) for x in y]) for y in scales]
        df_dict = {'Name':names, 'scale':str_scales, 'pair_ints':str_ints, 'all_ints':str_all_ints,
                  'Tuning':tunings, 'Culture':cultures, 'Continent':conts, 'Reference':ref, 'Theory':theory}
        new_dfs.append(pd.DataFrame(data=df_dict))

    return pd.concat(new_dfs, ignore_index=True)

def process_data():
    df_1, df_2, df_3, df_4 = load_data()
    df = reformat_data([df_1, df_2, df_3, df_4])
    df['n_notes'] = df.pair_ints.apply(lambda x: len(x.split(';')))

    ### Clean up duplicates
    to_bin = set()
    cultures = df.Culture.unique()
    for row in df.itertuples():
        if row[0] in to_bin:
            continue
        idx = df.loc[(df.Culture==row.Culture)&(df.pair_ints==row.pair_ints)].index
        if len(idx)>1:
            for i in idx[1:]:
                to_bin.add(i)
    df = df.drop(index=to_bin).reset_index(drop=True)
    return df
    

if __name__ == '__main__':

    df = process_data()

