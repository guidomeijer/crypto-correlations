#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 15:24:39 2021

@author: Guido Meijer
"""

import numpy as np
import pandas as pd
import datetime
from os import mkdir
from os.path import join, dirname, exists, realpath
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from Historic_Crypto import HistoricalData
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

# Settings
BIN_SIZE = 60  # seconds
TICKER = 'ETH'
NAME = 'Ethereum'
PLOT = False
MIN_R = 0.85
CACHE_DIR = '/media/guido/Data/AllenNeuropixel'

# Set directories
fig_dir = join(dirname(realpath(__file__)), 'exported_figs')
if not exists(fig_dir):
    mkdir(fig_dir)
this_fig_dir = join(fig_dir, NAME)
if not exists(this_fig_dir):
    mkdir(this_fig_dir)
data_dir = join(dirname(realpath(__file__)), 'data')
if not exists(data_dir):
    mkdir(data_dir)

# Query sessions
manifest_path = join(CACHE_DIR, 'manifest.json')
cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
sessions = cache.get_session_table()


def get_activity_vector(spike_times, bin_size):
    time_bins = np.arange(0, spike_times[-1], bin_size)
    spike_bins = np.empty(time_bins.shape[0])
    for i in range(len(time_bins[:-2])):
        spike_bins[i] = np.sum((spike_times > time_bins[i]) & (spike_times < time_bins[i + 1]))
    spike_bins = spike_bins[:-2]  # Exclude last time points (weird drop at the end of recording)
    activity_vector = spike_bins / bin_size
    return activity_vector


def get_crypto_vector(ticker, start_time):
    end_time = start_time + datetime.timedelta(hours=5)
    crypto_data = HistoricalData(f'{TICKER}-USD', BIN_SIZE,
                                 start_time.strftime("%Y-%m-%d-%H-%M"),
                                 end_time.strftime("%Y-%m-%d-%H-%M")).retrieve_data()
    return crypto_data['open'].values


# Loop through sessions
results_df = pd.DataFrame()
for i, session_id in enumerate(sessions.index.values):
    print(f'Processing session {session_id} [{i+1} of {sessions.index.values.shape[0]}]')
    session = cache.get_session_data(session_id)

    # Load in crypto data from the time of the recording
    crypto_vector = get_crypto_vector(TICKER, session.session_start_time)

    # Loop through units
    for j, unit_id in enumerate(session.units.index.values):

        # Correlate neuron activity with crypto value
        activity_vector = get_activity_vector(session.spike_times[unit_id], BIN_SIZE)
        # activity_vector = np.convolve(activity_vector, np.ones((3,))/3)
        r, p = pearsonr(activity_vector, crypto_vector[:activity_vector.shape[0]])

        # Add to results dataframe
        results_df = results_df.append(pd.DataFrame(index=[results_df.shape[0] + 1], data={
            'r': r, 'p': p, 'session_id': session_id, 'unit_id': unit_id,
            'subject': session.specimen_name,
            'acronym': session.units.loc[unit_id]['ecephys_structure_acronym']}))

        # Plot highly correlated neurons
        if (r > MIN_R) & PLOT:
            time_vector = np.linspace(0, activity_vector.shape[0]*BIN_SIZE, activity_vector.shape[0]) / 60
            sns.set_context('paper')
            f, ax1 = plt.subplots(1, 1, figsize=(5, 5), dpi=300)
            lns1 = ax1.plot(time_vector, activity_vector, color=sns.color_palette('colorblind')[0],
                            label='Firing rate (spikes/s)')
            ax1.set(xlabel='Time (min)', title=f'Single neuron activity versus {NAME} price')
            ax1.tick_params(axis='y', labelcolor=sns.color_palette('colorblind')[0])
            ax1.yaxis.label.set_color(sns.color_palette('colorblind')[0])
            ax2 = ax1.twinx()
            lns2 = ax2.plot(time_vector, crypto_vector[:activity_vector.shape[0]],
                            color=sns.color_palette('colorblind')[3],
                            label=f'Price of {NAME} (USD)')
            #ax2.set(ylabel=f'Value of {NAME} (USD)')
            ax2.tick_params(axis='y', labelcolor=sns.color_palette('colorblind')[3])
            ax2.yaxis.label.set_color(sns.color_palette('colorblind')[3])
            labs = [l.get_label() for l in lns1 + lns2]
            ax1.legend(lns1 + lns2, labs, frameon=False)
            sns.despine(ax=ax1, top=True, right=False, trim=True)
            sns.despine(ax=ax2, top=True, right=False, trim=True)
            plt.tight_layout()
            plt.savefig(join(this_fig_dir, f'{TICKER}_ses-{session_id}_unit-{unit_id}'))
            plt.close(f)

# Save results
results_df.to_csv(join(data_dir, f'{NAME}_correlations.csv'))
