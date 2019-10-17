import os
import sys

from itertools import product
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


BASE_DIR = '/home/jmcbride/Scales/Compared_data'
GRID_DIR = '/home/jmcbride/Scales/Processed_data'
REAL_DIR = '/home/jmcbride/Scales/Real_scales'

df_real = pd.read_feather(os.path.join(REAL_DIR, 'theories_real_scales.feather'))
df_real = df_real.loc[df_real.n_notes.apply(lambda x: x in [5,7])].reset_index(drop=True)

df_real.loc[df_real.Theory=='Y'].reset_index(drop=True).to_feather(os.path.join(REAL_DIR, 'Samples', 'theory.feather'))
df_real.loc[df_real.Theory=='N'].reset_index(drop=True).to_feather(os.path.join(REAL_DIR, 'Samples', 'instrument.feather'))

idx5 = df_real.loc[df_real.n_notes==5].index
idx7 = df_real.loc[df_real.n_notes==7].index

l5 = len(idx5)
l7 = len(idx7)

for frac in [0.4, 0.6, 0.8]:
    for i in range(10):
        idx = np.append(idx5[np.random.randint(l5, size=int(l5*frac))],
                        idx7[np.random.randint(l7, size=int(l7*frac))])
        df_real.loc[idx].reset_index(drop=True).to_feather(os.path.join(REAL_DIR, 'Samples', f'sample_f{frac:3.1f}_{i:02d}.feather'))
    




