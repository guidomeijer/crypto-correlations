#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 12:03:03 2021

@author: Guido Meijer
"""

from os import mkdir
from os.path import join, realpath, dirname, exists
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multitest import fdrcorrection

# Load in data
data_dir = join(dirname(realpath(__file__)), 'data')
btc_df = pd.read_csv(join(data_dir, 'Bitcoin_correlations.csv'))
eth_df = pd.read_csv(join(data_dir, 'Ethereum_correlations.csv'))
rand_df = pd.read_csv(join(data_dir, 'random_correlations.csv'))
controls_df = pd.read_csv(join(data_dir, 'Ethereum_controls.csv'))

# Perform false discorvery rate correction
fdr_sig_btc, _ = fdrcorrection(btc_df['p'])
fdr_sig_eth, _ = fdrcorrection(eth_df['p'])
fdr_sig_rand, _ = fdrcorrection(rand_df['p'])
fdr_sig_shuffle, _ = fdrcorrection(controls_df['p_shuffle'])
fdr_sig_permut, _ = fdrcorrection(controls_df['p_permutation'])
fdr_sig_linshift, _ = fdrcorrection(controls_df['p_linshift'])

# %% Plot
sns.set_context('talk')
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5), dpi=150, sharey=True)
ax1.bar(np.arange(3), [(np.sum(btc_df['p'] < 0.05) / btc_df.shape[0]) * 100,
                       (np.sum(eth_df['p'] < 0.05) / eth_df.shape[0]) * 100,
                       (np.sum(rand_df['p'] < 0.05) / rand_df.shape[0]) * 100],
        color=sns.color_palette('colorblind')[0])
ax1.plot([-0.75, 2.75], [5, 5], ls='--', color=[.5, .5, .5])
ax1.set(title='Significantly correlated neurons', ylabel='Percentage',
        ylim=[0, 80], xlim=[-0.75, 2.75],
        xticks=np.arange(3), xticklabels=['Bitcoin', 'Ethereum', 'Random'])

ax2.bar(np.arange(3), [(np.sum(fdr_sig_btc) / btc_df.shape[0]) * 100,
                       (np.sum(fdr_sig_eth) / eth_df.shape[0]) * 100,
                       (np.sum(fdr_sig_rand) / rand_df.shape[0]) * 100])
ax2.plot([-0.75, 2.75], [5, 5], ls='--', color=[.5, .5, .5])
ax2.set(title='False Discovery Rate (FDR) corrected', ylabel='', xlim=[-0.75, 2.75],
        xticks=np.arange(3), xticklabels=['Bitcoin', 'Ethereum', 'Random'])
ax2.yaxis.set_tick_params(labelbottom=True)

ax3.bar(np.arange(3), [(np.sum(btc_df['p'] < (0.05 / btc_df.shape[0])) / btc_df.shape[0]) * 100,
                       (np.sum(eth_df['p'] < (0.05 / btc_df.shape[0])) / eth_df.shape[0]) * 100,
                       (np.sum(rand_df['p'] < (0.05 / btc_df.shape[0])) / rand_df.shape[0]) * 100])
ax3.plot([-0.75, 2.75], [5, 5], ls='--', color=[.5, .5, .5])
ax3.set(title='Bonferroni corrected', ylabel='', xlim=[-0.75, 2.75],
        xticks=np.arange(3), xticklabels=['Bitcoin', 'Ethereum', 'Random'])
ax3.yaxis.set_tick_params(labelbottom=True)
sns.despine(trim=False)

# Save figure
fig_dir = join(dirname(realpath(__file__)), 'exported_figs')
if not exists(fig_dir):
    mkdir(fig_dir)
plt.savefig(join(fig_dir, 'percentage_significant_neurons'))

sns.set_context('talk')
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
ax1.hist(btc_df['r'], histtype='step', facecolor=sns.color_palette('colorblind')[0],
         label='Bitcoin', lw=2, bins=25)
ax1.hist(eth_df['r'], histtype='step', facecolor=sns.color_palette('colorblind')[1],
         label='Ethereum', lw=2, bins=25)
ax1.hist(rand_df['r'], histtype='step', facecolor=sns.color_palette('colorblind')[2],
         label='Random', lw=2, bins=25)
ax1.set(title='Pearson correlation coefficient (r)', ylabel='Neuron count', ylim=[0, 6000])
ax1.legend(frameon=False)

ax2.hist(btc_df['p'], histtype='step', facecolor=sns.color_palette('colorblind')[0],
         label='Bitcoin', lw=2, bins=25)
ax2.hist(eth_df['p'], histtype='step', facecolor=sns.color_palette('colorblind')[1],
         label='Ethereum', lw=2, bins=25)
ax2.hist(rand_df['p'], histtype='step', facecolor=sns.color_palette('colorblind')[2],
         label='Random', lw=2, bins=25)
ax2.set(title='Pearson correlation (p-values)', ylabel='Neuron count', ylim=[0, 30000])
ax2.legend(frameon=False)

plt.tight_layout()
sns.despine(trim=True)
plt.savefig(join(fig_dir, 'histogram_correlation'))

sns.set_context('talk')
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5), dpi=150, sharey=False)
ax1.bar(np.arange(3), [(np.sum(controls_df['p_shuffle'] < 0.05) / controls_df.shape[0]) * 100,
                       (np.sum(controls_df['p_permutation'] < 0.05) / controls_df.shape[0]) * 100,
                       (np.sum(controls_df['p_linshift'] < 0.05) / controls_df.shape[0]) * 100],
        color=sns.color_palette('colorblind')[0])
ax1.plot([-0.75, 2.75], [5, 5], ls='--', color=[.5, .5, .5])
ax1.set(title='Significantly correlated neurons', ylabel='Percentage',
        ylim=[0, 50], xlim=[-0.75, 2.75],
        xticks=np.arange(3), xticklabels=['Shuffle', 'Permutation', 'Linshift'])

ax2.hist(controls_df['p_shuffle'] , histtype='step', facecolor=sns.color_palette('colorblind')[0],
         label='Shuffle', lw=2, bins=25)
ax2.hist(controls_df['p_permutation'] , histtype='step', facecolor=sns.color_palette('colorblind')[1],
         label='Permutation', lw=2, bins=25)
ax2.hist(controls_df['p_linshift'] , histtype='step', facecolor=sns.color_palette('colorblind')[2],
         label='Linshift', lw=2, bins=25)
ax2.set(title='Pearson correlation (p-values)', ylabel='Neuron count', ylim=[0, 20000])
ax2.legend(frameon=False)

ax3.bar(np.arange(3), [(np.sum(fdr_sig_shuffle) / fdr_sig_shuffle.shape[0]) * 100,
                       (np.sum(fdr_sig_permut) / fdr_sig_permut.shape[0]) * 100,
                       (np.sum(fdr_sig_linshift) / fdr_sig_linshift.shape[0]) * 100],
        color=sns.color_palette('colorblind')[0])
ax3.plot([-0.75, 2.75], [5, 5], ls='--', color=[.5, .5, .5])
ax3.set(title='FDR-corrected significant neurons', ylabel='Percentage',
        ylim=[0, 50], xlim=[-0.75, 2.75],
        xticks=np.arange(3), xticklabels=['Shuffle', 'Permutation', 'Linshift'])

plt.tight_layout()
sns.despine(trim=False)
plt.savefig(join(fig_dir, 'control_correlation'))
