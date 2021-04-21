#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 15:24:39 2021

@author: Guido Meijer
"""

import numpy as np
import pandas as pd
import datetime
import random
from os import mkdir
from os.path import join, dirname, exists, realpath
from scipy.stats import pearsonr
from Historic_Crypto import HistoricalData
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

# Settings
BIN_SIZE = 60  # seconds
TICKER = 'ETH'
NAME = 'Ethereum'
SHUFFLES = 1000
CACHE_DIR = '/media/guido/Data/AllenNeuropixel'

# Set directories
data_dir = join(dirname(realpath(__file__)), 'data')
if not exists(data_dir):
    mkdir(data_dir)

# Query sessions
manifest_path = join(CACHE_DIR, 'manifest.json')
cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
sessions = cache.get_session_table()

# Load in null vectors
null_crypto = np.load(join(data_dir, f'{TICKER}_null_vectors.npy'))


def get_activity_vector(spike_times, bin_size):
    spike_times = spike_times[~np.isnan(spike_times)]  # get rid of some weird spikes
    spike_times = spike_times[(spike_times > 0.1) & (spike_times < 18000)]
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


def sigtest_linshift(X, y, fStatMeas, D=300):
    """
    Original code by Brandon Benson, modified for this purpose
    Uses a provably conservative Linear Shift technique (Harris, Kenneth Arxiv 2021,
    https://arxiv.org/ftp/arxiv/papers/2012/2012.06862.pdf) to estimate
    significance level of a statistical measure. fStatMeas computes a
    scalar statistical measure (e.g. R^2) from the data matrix, X, and the variable, y.
    A central window of X and y of size, D, is linearly shifted to generate a null distribution
    of statistical measures.  Significance level is reported relative to this null distribution.

    X : 2-d array
        Data of size (elements, timetrials)
    y : 1-d array
        predicted variable of size (timetrials)
    fStatMeas : function
        takes arguments (X, y) and returns a scalar statistical measure of how well X decodes y
    D : int
        the window length along the center of y used to compute the statistical measure.
        must have room to shift both right and left: len(y) >= D+2

    Returns
    -------
    alpha : conservative p-value e.g. at a significance level of b, if alpha <= b then reject the
            null hypothesis.
    statms_real : the value of the statistical measure evaluated on X and y
    statms_pseuds : a 1-d array of statistical measures evaluated on shifted versions of y
    """
    assert len(y) >= D + 2

    T = len(y)
    N = int((T - D) / 2)

    shifts = np.arange(-N, N + 1)

    # compute all statms
    statms_real = fStatMeas(X[N:T - N], y[N:T - N])[0]
    statms_pseuds = np.zeros(len(shifts))
    for si in range(len(shifts)):
        s = shifts[si]
        statms_pseuds[si] = fStatMeas(np.copy(X[N:T - N]), np.copy(y[s + N:s + T - N]))[0]

    M = np.sum(statms_pseuds >= statms_real)
    alpha = M / len(statms_pseuds)

    return alpha, statms_real, statms_pseuds


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
        r, p = pearsonr(activity_vector, crypto_vector[:activity_vector.shape[0]])

        # Get p-value by shuffling
        r_shuffle = np.empty(SHUFFLES)
        for k in range(SHUFFLES):
            activity_random = activity_vector.copy()
            random.shuffle(activity_random)
            r_shuffle[k], _ = pearsonr(activity_random, crypto_vector[:activity_vector.shape[0]])
        p_shuffle = np.sum(r > r_shuffle) / r_shuffle.shape[0]

        # Correlate all null vectors
        r_null = np.empty(null_crypto.shape[0])
        for k in range(null_crypto.shape[0]):
            r_null[k], _ = pearsonr(activity_vector, null_crypto[k, :activity_vector.shape[0]])

        # Define p-value as fraction of null distribution
        p_permutation = np.sum(r > r_null) / r_null.shape[0]

        # Get p-value from linshift method
        p_linshift = sigtest_linshift(activity_vector, crypto_vector[:activity_vector.shape[0]],
                                      pearsonr, D=50)[0]

        # Add to results dataframe
        results_df = results_df.append(pd.DataFrame(index=[results_df.shape[0] + 1], data={
            'r': r, 'p': p, 'p_permutation': p_permutation, 'p_linshift': p_linshift,
            'p_shuffle': p_shuffle,
            'session_id': session_id, 'unit_id': unit_id,
            'subject': session.specimen_name}))

# Save results
results_df.to_csv(join(data_dir, f'{NAME}_controls.csv'))
