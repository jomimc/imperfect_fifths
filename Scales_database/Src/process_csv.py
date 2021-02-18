import os
from pathlib import Path

import numpy as np
import pandas as pd

import utils

PATH_BASE = [p for p in [Path.cwd()] + list(Path.cwd().parents) if p.name == 'imperfect_fifths'][0]
DATA_DIR = PATH_BASE.joinpath("Scales_database", "Data")

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

    # Data from Ellis (1885)
    df_5 = pd.read_csv(os.path.join(DATA_DIR,'scales_E.csv'))
    df_5 = utils.reformat_original_csv_data(df_5)

    return df_1, df_2, df_3, df_4, df_5


### Create new DataFrame with intervals in cents for all tunings

def reformat_data(old_dfs, oct_cut=50):
    new_dfs = []

    for i, df in enumerate(old_dfs):
        if i==0:
            df_dict = utils.extract_scales_and_ints_from_scales(df)
            df_dict['pair_ints'] = [';'.join([str(int(round(x))) for x in y]) for y in df_dict['pair_ints']]
        elif i==3:
            cols = ['Culture', 'Tuning', 'Continent', 'Country', 'Name', 'pair_ints','Reference', 'RefID', 'Theory']
            df_dict = {c:df[c] for c in cols}
            df_dict['scale'] = [[0] + list(np.cumsum([int(x) for x in ints.split(';')])) for ints in df_dict['pair_ints']]
            df_dict['all_ints'] = [[s[i] - s[j] for j in range(len(s)) for i in range(j+1,len(s))] for s in df_dict['scale']]
        else:
            df_dict = utils.extract_scales_and_ints_from_unique(df, oct_cut=oct_cut)
            df_dict['pair_ints'] = [';'.join([str(int(round(x))) for x in y]) for y in df_dict['pair_ints']]

        ### all_ints only counts intervals that fall within a single octave
        df_dict['all_ints'] = [';'.join([str(int(round(x))) for x in y]) for y in df_dict['all_ints']]
        df_dict['scale'] = [';'.join([str(int(round(x))) for x in y]) for y in df_dict['scale']]
        new_dfs.append(pd.DataFrame(data=df_dict))

    return pd.concat(new_dfs, ignore_index=True)

def process_data(oct_cut=50):
    df_list = load_data()
    df = reformat_data(df_list, oct_cut=oct_cut)
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

    ### Only include scales with 4 <= N <= 9
    df = df.loc[(df.n_notes>=4)&(df.n_notes<=9)].reset_index(drop=True)

    ### Some basic metrics for scales
    df['min_int'] = df.pair_ints.apply(lambda x: min(utils.str_to_ints(x)))
    df['max_int'] = df.pair_ints.apply(lambda x: max(utils.str_to_ints(x)))
    df['octave'] = df.scale.apply(lambda x: max(utils.str_to_ints(x)))
    df['irange'] = df['max_int'] - df['min_int']

    ### all_ints2 counts intervals that are smaller than an octave,
    ### but can be made within the first two octaves
    ### e.g. the interval between the 2nd note and the 8th (1st) note in a 7-note scale
    df = utils.get_all_ints(df)

    ### Clustering the scales by similarity between adjacent interval sets
    df = utils.label_scales_by_cluster(df)
    return df
    

if __name__ == '__main__':

    df = process_data()
    df.to_pickle("../Data_processed/real_scales.pickle")


