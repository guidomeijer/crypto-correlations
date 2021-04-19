#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 14:48:03 2021

@author: Guido Meijer
"""

from os import mkdir
from os.path import join, realpath, dirname, exists
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ibllib.atlas import BrainRegions

# Settings
MIN_UNITS = 10


def get_full_region_name(acronyms):
    brainregions = BrainRegions()
    full_region_names = []
    for i, acronym in enumerate(acronyms):
        try:
            regname = brainregions.name[np.argwhere(brainregions.acronym == acronym).flatten()][0]
            full_region_names.append(regname)
        except IndexError:
            full_region_names.append(acronym)
    if len(full_region_names) == 1:
        return full_region_names[0]
    else:
        return full_region_names


# Load in data (run correlate_crypto and correlate_random first)
data_dir = join(dirname(realpath(__file__)), 'data')
btc_df = pd.read_csv(join(data_dir, 'Bitcoin_correlations.csv'))
eth_df = pd.read_csv(join(data_dir, 'Ethereum_correlations.csv'))

# Restructure dataframe
btc_df['sig'] = btc_df['p'] < 0.05
btc_regions = (btc_df.groupby('acronym').sum()['sig'] / btc_df.groupby('acronym').size()) * 100
btc_regions = btc_regions[btc_df.groupby('acronym').size() > MIN_UNITS]
btc_regions = btc_regions.to_frame().reset_index()
btc_regions = btc_regions.rename(columns={0: 'perc'})
btc_regions['region'] = get_full_region_name(btc_regions['acronym'])
btc_regions = btc_regions[btc_regions['acronym'] != 'grey']

# %% Plot

sns.set_context('paper')
f, ax1 = plt.subplots(1, 1, figsize=(12, 8), dpi=300)
sns.barplot(x='perc', y='region', data=btc_regions, color=sns.color_palette('colorblind')[0])
ax1.set(xlabel='Percentage of significant neurons', xlim=[0, 100], ylabel='')
plt.tight_layout()
sns.despine(trim=False)

fig_dir = join(dirname(realpath(__file__)), 'exported_figs')
if not exists(fig_dir):
    mkdir(fig_dir)
plt.savefig(join(fig_dir, 'regions_significant_neurons'))
