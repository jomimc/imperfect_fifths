import os

import numpy as np
import pandas as pd


CLEAN_DIR = "/home/johnmcbride/projects/Scales/imperfect_fifths/Results/Processed"

if 0:

    df  = pd.read_feather(os.path.join(CLEAN_DIR, "monte_carlo_results_1.feather"))

    for i in df.loc[df.mfm_10.notnull()].index:
        f = df.loc[i, 'fName']
        print(i, f)
        fname = os.path.split(f)[1]
        base = '/media/johnmcbride/961391f1-186f-4345-881e-92d8bb3931c8/Projects/Scales/Results/Toy_model_scales/Old_Processed'
        tmp1 = pd.read_feather(f)
        tmp2 = pd.read_feather(os.path.join(base, fname))
        tmp1['mix_scale'] = tmp2['mix_scale']
        tmp1.to_feather(f)



df2 = pd.read_feather(os.path.join(CLEAN_DIR, "monte_carlo_results_2.feather"))


for i in df2.loc[df2.mfm_10.notnull()].index:
    f = df2.loc[i, 'fName']
    print(i, f)
    fname = os.path.split(f)[1]
    base = '/media/johnmcbride/961391f1-186f-4345-881e-92d8bb3931c8/Projects/Scales/Results/Toy_model_scales/Old_Processed3'
    tmp1 = pd.read_feather(f)
    tmp2 = pd.read_feather(os.path.join(base, fname))
    tmp1['mix_scale'] = tmp2['mix_scale']
    tmp1.to_feather(f)



