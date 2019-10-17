import argparse
import glob
import os
import pickle
import sys
import time

from itertools import product
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.nonparametric.api as smnp
import swifter

import utils
import graphs

N_PROC = 10

BASE_DIR = '/home/johnmcbride/projects/Scales/Data_compare/'
RAW_DIR  = '/home/johnmcbride/projects/Scales/Toy_model/Data/Raw/'
PRO_DIR  = '/home/johnmcbride/projects/Scales/Toy_model/Data/Processed/'
REAL_DIR = os.path.join(BASE_DIR, 'Processed/Real', 'Samples')
DIST_DIR = os.path.join(BASE_DIR, 'Processed/Real', 'Sample_dist')


def calc_relative_entropy(pk, qk):
    RE = 0.0
    for i in range(len(pk)):
        if pk[i] <= 0 or qk[i] <= 0:
            pass
        else:
            RE += pk[i] * np.log2(pk[i] / qk[i])
    return RE

def calc_jensen_shannon_distance(pk, qk):
    mk = 0.5 * (pk + qk)
    return (0.5 * (calc_relative_entropy(pk, mk) + calc_relative_entropy(qk, mk))) ** 0.5


def smooth_dist_kde(df, cat='pair_ints', hist=False, nbins=1202):
    X = [float(x) for y in df.loc[:,cat] for x in y.split(';')]
    kde = smnp.KDEUnivariate(np.array(X))
    kde.fit(kernel='gau', bw='scott', fft=1, gridsize=10000, cut=20)
    grid = np.linspace(0, 1200, num=nbins-1)
    y = np.array([kde.evaluate(x) for x in grid]).reshape(nbins-1)
    if hist:
        xtra = (nbins-2)/1200./2.
        bins = np.linspace(-xtra, 1200+xtra, num=nbins)
        hist, bins = np.histogram(X, bins=bins, normed=True)
        return grid, y, hist
    else:
        return grid, y


def get_KDE(df, cat):
    xKDE, yKDE = smooth_dist_kde(df, cat=cat)
    return yKDE / np.trapz(yKDE)


def get_dists_file(s, cat='pair_ints', nbins=1202):
    out = {}
    if not os.path.exists(os.path.join(DIST_DIR, f"{s}_n7_hist.npy")):
        df = pd.read_feather(os.path.join(REAL_DIR, f"{s}.feather"))
    for n in [5,7]:
        fHist = os.path.join(DIST_DIR, f"{s}_{cat}_n{n}_hist.npy")
        fKDE  = os.path.join(DIST_DIR, f"{s}_{cat}_n{n}_kde.npy")
        if os.path.exists(fHist):
            X, hist = np.load(fHist)
            X, kde = np.load(fKDE)
        else:
            X, kde, hist = smooth_dist_kde(df.loc[df.n_notes==n], cat=cat, hist=True, nbins=nbins)
            np.save(fHist, np.array([X, hist]))
            np.save(fKDE, np.array([X, kde]))
        out[n] = [X, kde, hist]
    return out


def how_much_real_scales_predicted(df, n_real, w, s):
#   try:
        return float(len(set([int(x) for y in df[f"{s}_w{w:02d}"] for x in y.split(';') if len(y)]))) / float(n_real)
#   except:
#       return None


def rename_processed_files(f, s='sample_'):
    root, fName = os.path.split(f)
    return os.path.join(root, f"{s}{fName}")


def load_model_filenames():
    paths = pickle.load(open(os.path.join(BASE_DIR, 'best_models.pickle'), 'rb'))
    return [rename_processed_files(paths[k][n]) for k, n in product(paths.keys(), [5,7])]


def calculate_metrics(y1, y2):
    y1 = y1.reshape(y1.size)
    y2 = y2.reshape(y2.size)

    err_sq = np.sqrt(np.dot(y1-y2, y1-y2))

    d1 = y1[1:] - y1[:-1]
    d2 = y2[1:] - y2[:-1]
    deriv_es = np.sqrt(np.dot(d1-d2, d1-d2))

    return [err_sq, deriv_es, (err_sq * deriv_es)**0.5]


def scale_rsq(Y1, Y2):
    SStot = np.sum((Y1 - np.mean(Y1))**2)
    SSres = np.sum((Y1 - Y2)**2)
    return  1 - SSres/SStot


if __name__ == "__main__":

    timeS = time.time()

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--partabase', action='store', default='None', type=str)
    args = parser.parse_args()

    categories = ['pair_ints', 'scale']
    n_arr = np.arange(4,10,dtype=int)

    samples = ['theory', 'instrument'] + [f"sample_f{frac:3.1f}_{i:02d}" for frac in [0.4, 0.6, 0.8] for i in range(10)]
    files = [f"{s}.feather" for s in samples]

    int_dists = [get_dists_file(s) for s in samples]
    hist_dists = [get_dists_file(s, cat='scale', nbins=42) for s in samples]

#   print(f"Real scales loaded after {(time.time()-timeS)/60.} minutes")


    pro_files = load_model_filenames()

    def extract_stats_each_model(fName):
        df = pd.read_feather(fName)
        bits = os.path.split(fName)[1].split('_')
        n = int(bits[1].strip('n'))
        idx = [i for i in range(len(bits)) if bits[i][0]=='M'][0]
        bias = '_'.join(bits[2:idx])
        mi = int(bits[idx].strip('MI'))
        ma = int(bits[idx+1].strip('MA'))
        beta = float(bits[-1].strip('.feather'))

        n_sample = df.n_att.sum()
        q = float(len(df))/float(n_sample)

        output = [n, mi, ma, bias, beta, q, n_sample]

        X, iKDE, iHist = smooth_dist_kde(df, cat='pair_ints', hist=True)
        X, sKDE, sHist = smooth_dist_kde(df, cat='scale', hist=True, nbins=42)

        for i, f in enumerate(files):
            df_real = pd.read_feather(os.path.join(REAL_DIR, f))
            n_real = len(df_real.loc[df_real.n_notes==n])
            frac_real = [how_much_real_scales_predicted(df, n_real, w, f'{samples[i]}_ss') for w in [10, 20]]

            metrics = calculate_metrics(int_dists[i][n][1], iKDE)
            scale_R2 = scale_rsq(sHist,hist_dists[i][n][2])

            output.extend([n_real] + frac_real + metrics + [scale_R2])

        return output + [fName]

    biases = ['none',
             'distI_1_0', 'distI_2_0', 'distI_3_0', 'distI_0_1', 'distI_0_2',
             'distI_1_1', 'distI_2_1', 'distI_1_2', 'distI_2_2',
             'opt_c', 'opt_c_I1', 'opt_c_I2', 'opt_c_s2', 'opt_c_s3'] + \
             [f"hs_n{i}_w{w:02d}" for i in range(1,4) for w in [5,10,15,20]] + \
             [f"hs_r3_w{w:02d}" for w in [5,10,15,20]] + \
             [f"ahs{i:02d}_w{w:02d}" for i in range(1,11) for w in [5,10,15,20]] + \
             [f"im5_r{r:3.1f}_w{w:02d}" for r in [0, 0.5, 1, 2] for w in [5,10,15,20]] + \
             [f"Nhs_n1_w{w:02d}" for w in [5,10,15,20]] + \
             [f"Nhs_n2_w{w:02d}" for w in [5,10,15,20]] + \
             [f"Nhs_n3_w{w:02d}" for w in [5,10,15,20]] + \
             [f"Nim5_r0.0_w{w:02d}" for w in [5,10,15,20]] + \
             [f"TRANSB_{i}" for i in [1,2,3]] + \
             [f"TRANS{a}_{b}" for a in ['A', 'B'] for b in range(1,4)] + \
             [f"HAR_{b}_{a}" for a in range(1,4) for b in range(5,25,5)] + \
             [f"{a}_{b}" for a in ['HAR', 'FIF'] for b in range(5,25,5)]

#            ['hs_r3_w05', 'hs_r3_w10', 'hs_r3_w15', 'hs_r3_w20'] + \
#            [f"im5_r0.75_w{w:02d}" for w in [5,10,15,20] + 

    groups = ['none'] + ['distI']*3 + ['S#1']*2 + ['distI_S#1']*4 + \
             ['distW'] + ['distW_S#1']*2 + ['distW_S#2']*2 + ['HS']*12 + ['im5']*4 + ['AHS']*40 + ['im5']*16 + \
             ['HS']*12 + ['im5']*4 + ['TRANSB']*3  + \
             ['TRANS']*6 + ['HAR']*4 + ['HAR2']*4 + ['HAR3']*4 + ['HAR']*4 + ['FIF']*4
    bias_groups = {biases[i]:groups[i] for i in range(len(biases))} 


    with mp.Pool(N_PROC) as pool:
        results = list(pool.imap_unordered(extract_stats_each_model, pro_files))

    print(f"Model comparison finished after {(time.time()-timeS)/60.} minutes")
    df = pd.DataFrame(columns=['n_notes', 'min_int', 'max_int', 'bias', 'beta', 'quantile', 'n_sample'] + \
                              [f"{s}_{a}" for s in samples for a in ['n_real', 'fr_10', 'fr_20', 'RMSD', 'dRMSD', 'met1', 'sRMSD']] + \
                              ['fName'], data=results)



    df['bias_group'] = df.bias.apply(lambda x: bias_groups[x])
    df['logq'] = np.log10(df['quantile'])
    df = graphs.rename_bias_groups(df)
    df = graphs.rename_biases(df)
    

    print(f"DataFrame compiled after {(time.time()-timeS)/60.} minutes")

    if args.partabase == 'None':
        df.to_feather(os.path.join(BASE_DIR, 'Processed', 'database_sensitivity.feather'))



