import os
from pathlib import Path

import numpy as np
import pandas as pd

import utils

PATH_BASE = [p for p in [Path.cwd()] + list(Path.cwd().parents) if p.name == 'imperfect_fifths'][0]
DATA_DIR = PATH_BASE.joinpath("Scales_database", "Data")

### Load the data from csv files
def load_data_old():
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


def load_raw_data():
    return [pd.read_excel('../scales_database.xlsx', f"scales_{a}") for a in 'ABCDEF']


### Create new DataFrame with intervals in cents for all tunings

def convert_raw_data_to_scales(old_dfs, oct_cut=50, use_mode=False):
    new_dfs = []

    for i, df in enumerate(old_dfs):
        if i == 0:
            new_dfs.append(utils.match_scales_to_tunings(df))
        else:
            new_dfs.append(utils.extract_scales_from_measurements(df, oct_cut=oct_cut, use_mode=use_mode))

    return pd.concat(new_dfs, ignore_index=True)


def same_ints(i1, i2):
    if len(i1) != len(i2):
        return False
    return np.all(i1 == i2)
    

### Remove any duplicates within the same Culture
#       some may have been added due to reports in secondary sources
#       OR some scales are labelled differently in reports due
#       to starting on different funadmental frequencies
#       OR some scales may simply be duplicates due to chance
def remove_duplicates(df):
    to_bin = set()
    cultures = df.Culture.unique()
    for row in df.itertuples():
        if row[0] in to_bin:
            continue
        idx = (df.Culture==row.Culture) & (df.adj_ints.apply(lambda x: same_ints(x, row.adj_ints)))
        if sum(idx)>1:
            for i in np.where(idx)[0][1:]:
                to_bin.add(i)
    return df.drop(index=to_bin).reset_index(drop=True)



def process_data(oct_cut=50, use_mode=False):
    df_list = load_raw_data()
    df = convert_raw_data_to_scales(df_list, oct_cut=oct_cut, use_mode=use_mode)
    df = df.rename(columns={'Intervals':'adj_ints'})
    df = remove_duplicates(df)

    ### Only include scales with 4 <= N <= 9
    df = df.loc[(df.n_notes>=4)&(df.n_notes<=9)].reset_index(drop=True)

    ### Some basic metrics for scales
    df['min_int'] = df.adj_ints.apply(min)
    df['max_int'] = df.adj_ints.apply(max)
    df['octave'] = df.scale.apply(max)
    df['octave_dev'] = np.abs(df['octave'] - 1200)
    df['irange'] = df['max_int'] - df['min_int']

#   ### Clustering the scales by similarity between adjacent interval sets
#   df = utils.label_scales_by_cluster(df)
    return df
    

if __name__ == '__main__':

    df = process_data()
    df.to_pickle("../Data_processed/real_scales.pickle")


