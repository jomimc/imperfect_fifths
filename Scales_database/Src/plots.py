from pathlib import Path
import pickle

import geopandas
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import numpy as np
from palettable.colorbrewer.qualitative import Paired_12, Set2_8, Dark2_8, Pastel2_8, Pastel1_9
from palettable.scientific.diverging import Berlin_20
from palettable.scientific.sequential import Bamako_4

import pandas as pd
import seaborn as sns
from shapely.geometry.point import Point



def get_color(n_notes):
    if len(n_notes):
        return Bamako_4.mpl_colors[7-int(np.mean(n_notes))]
    else:
        return (0.6, 0.6, 0.6)

def get_n_notes(n_notes):
    if len(n_notes):
        return np.mean(n_notes)
    else:
        return 7

def world_map(df):
    df = df.loc[(df.n_notes>3)&(df.n_notes<10)].reset_index(drop=True)
#   df.loc[df.Country=='Laos', 'Country'] = "Lao PDR"
    df.loc[df.Country=='Singapore', 'Country'] = "Malaysia"
    df.loc[df.Country=='Korea', 'Country'] = "South Korea"

    counts = df.loc[(df.Theory=='N')&(df.Country.str.len()>0), 'Country'].value_counts()
    countries = np.array(counts.keys())
    co = counts.values
    idx = np.argsort(countries)
    countries = countries[idx]
    co = co[idx]

    world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
    world['cent_col'] = world.centroid.values
    world['col'] = [get_color(df.loc[df.Country==c, 'n_notes'].values) for c in world.name]
    world['n_notes'] = [get_n_notes(df.loc[df.Country==c, 'n_notes'].values) for c in world.name]

    coord = [world.loc[world.name==c, 'cent_col'].values[0] for c in countries]
    n_notes = [df.loc[df.Country==c, 'n_notes'].mean() for c in countries]
    cols = [Bamako_4.mpl_colors[7-int(n)] for n in n_notes]
    gdf = geopandas.GeoDataFrame(pd.DataFrame(data={'Country':countries, 'count':co, 'coord':coord, 'n_notes':n_notes, 'col':cols}), geometry='coord')
    

#   Cont = ['Western', 'Middle East', 'South Asia', 'East Asia', 'South East Asia', 'Africa', 'Oceania', 'Latin America']
#   theory = [len(df.loc[(df.Theory=='Y')&(df.Region==c)]) for c in Cont]
#   inst   = [len(df.loc[(df.Theory=='N')&(df.Region==c)]) for c in Cont]
#   n_notes = [df.loc[df.Region==c, 'n_notes'].mean() for c in Cont]
#   cols = [Bamako_4.mpl_colors[7-int(n)] for n in n_notes]
#   cmap = Berlin_20.mpl_colormap

#   cont_coord = [Point(*x) for x in [[17, 48], [32, 33], [79, 24], [110, 32], [107, 12], [18, 8], [150, -20], [-70, -10]]]

#   cont_df = geopandas.GeoDataFrame(pd.DataFrame(data={'Cont':Cont, 'count':theory, 'coord':cont_coord, 'n_notes':n_notes, 'col':cols}), geometry='coord')

    fig, ax = plt.subplots()
    col = Paired_12.mpl_colors
    ft1 = 12

    world.plot(ax=ax, column='n_notes', edgecolor=(1.0,1.0,1.0), lw=0.2,
               legend=True, legend_kwds={'label':"Mean scale degree", "orientation":"horizontal"})
    world.loc[world.name.apply(lambda x: x not in countries)].plot(ax=ax, color=(0.6, 0.6, 0.6), edgecolor=(1.0,1.0,1.0), lw=0.2)

#   world.plot(ax=ax, color=(0.6, 0.6, 0.6), edgecolor=(1.0,1.0,1.0), lw=0.2)
#   world.loc[world.name.apply(lambda x: x in countries)].plot(ax=ax, color=(0.3, 0.3, 0.3), edgecolor=(1.0,1.0,1.0), lw=0.2)
#   for c1, c2 in zip(countries, cols):
#       world.loc[world.name==c1].plot(ax=ax, color=c2, edgecolor=(1.0,1.0,1.0), lw=0.2)
#   gdf.plot(color='r', ax=ax, markersize=gdf['count'].values*0.5, alpha=1)
#   cont_df.plot(color='g', ax=ax, markersize=cont_df['count'].values)
    
#   gdf.plot(color=gdf['count'].values, ax=ax, alpha=1)
#   cont_df.plot(color='g', ax=ax, markersize=cont_df['count'].values)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-185, 185)
    ax.set_ylim(-60, 88)

