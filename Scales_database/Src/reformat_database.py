import os

import numpy as np
import pandas as pd

import utils

DATA_DIR = '/home/johnmcbride/projects/Scales/imperfect_fifths/Scales_database/Data_processed/'


def load_database():
    return pd.read_pickle(os.path.join(DATA_DIR, "real_scales.pickle"))


def reformat_database(df):
    cols = ['Name', 'Culture', 'Country', 'Region', "Theory", "Reference", "RefID"]
    df_new = df.loc[:, cols]
    for i in df.index:
        ints = {j:x for j, x in enumerate(utils.str_to_ints(df.loc[i, 'pair_ints']))}
        scale = {j:x for j, x in enumerate(utils.str_to_ints(df.loc[i, 'scale'])[1:])}
        for j in range(9): 
            df_new.loc[i, f"int_{j+1}"] = ints.get(j, '')
        for j in range(9): 
            df_new.loc[i, f"scale_{j+1}"] = scale.get(j, '')
    return df_new

def save_database(df):
    df = df.sort_values(by=["Theory", "Region", "Culture", "Name"])
    df.to_csv("database_format4sharing.csv", index=False)


if __name__ == "__main__":

    save_database(reformat_database(load_database()))
        

