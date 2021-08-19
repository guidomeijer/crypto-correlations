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
from ibllib.atlas import BrainRegions
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

# Settings
BIN_SIZE = 60  # granularity of crypto trace in seconds (min is 60)
N_TRIALS = 500  # number of pseudo trials
TRIAL_LENGTH = 0.5  # trial length in seconds
TICKER = 'ETH'
NAME = 'Ethereum'
PLOT = True
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


def get_activity_trials(spike_times, crypto_vector, crypto_time, n_trials=500, trial_length=0.3):
    spike_times = spike_times[~np.isnan(spike_times)]  # get rid of some weird spikes
    spike_times = spike_times[(spike_times > 0.1) & (spike_times < 18000)]
    trial_onsets = np.sort(np.random.uniform(spike_times[0], spike_times[-1], size=n_trials))
    trial_spikes, trial_crypto = np.empty(n_trials), np.empty(n_trials)
    for i, onset in enumerate(trial_onsets):
        trial_spikes[i] = np.sum((spike_times > onset) & (spike_times < onset + trial_length))
        trial_crypto[i] = crypto_vector[np.argmin(np.abs(crypto_time - onset))]
    return trial_spikes, trial_crypto


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
    crypto_time = np.arange(0, crypto_vector.shape[0] * BIN_SIZE, BIN_SIZE)

    # Get full region names
    these_units = session.units

    # Loop through units
    for j, unit_id in enumerate(these_units.index.values):

        # Correlate neuron activity with crypto value
        trial_spikes, trial_crypto = get_activity_trials(session.spike_times[unit_id], crypto_vector,
                                                         crypto_time, N_TRIALS, TRIAL_LENGTH)
        r, p = pearsonr(trial_spikes, trial_crypto)

        # Add to results dataframe
        results_df = results_df.append(pd.DataFrame(index=[results_df.shape[0] + 1], data={
            'r': r, 'p': p, 'session_id': session_id, 'unit_id': unit_id,
            'subject': session.specimen_name,
            'acronym': these_units.loc[unit_id]['ecephys_structure_acronym']}))

# Save results
results_df.to_csv(join(data_dir, f'{NAME}_correlations_trials.csv'))
