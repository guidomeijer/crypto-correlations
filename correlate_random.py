#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 09:55:12 2021

@author: Guido Meijer
"""

import numpy as np
import pandas as pd
from os import mkdir
from os.path import join, dirname, exists, realpath
from scipy.stats import pearsonr
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

# Settings
BIN_SIZE = 60  # seconds
CACHE_DIR = '/media/guido/Data/AllenNeuropixel'

# Set directories
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


# Loop through sessions
results_df = pd.DataFrame()
for i, session_id in enumerate(sessions.index.values):
    print(f'Processing session {session_id} [{i+1} of {sessions.index.values.shape[0]}]')
    session = cache.get_session_data(session_id)

    # Loop through units
    for j, unit_id in enumerate(session.units.index.values):

        # Correlate neuron activity with random vector
        activity_vector = get_activity_vector(session.spike_times[unit_id], BIN_SIZE)
        random_vector = np.random.rand(activity_vector.shape[0])
        r, p = pearsonr(activity_vector, random_vector)

        # Add to results dataframe
        results_df = results_df.append(pd.DataFrame(index=[results_df.shape[0] + 1], data={
            'r': r, 'p': p, 'session_id': session_id, 'unit_id': unit_id,
            'subject': session.specimen_name,
            'acronym': session.units.loc[unit_id]['ecephys_structure_acronym']}))

# Save results
results_df.to_csv(join(data_dir, 'random_correlations.csv'))
