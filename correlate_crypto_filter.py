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
from scipy.signal import filtfilt, butter
from Historic_Crypto import HistoricalData
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

# Settings
BIN_SIZE = 60  # seconds
TICKER = 'BTC'
NAME = 'Bitcoin'
FILT = {'high_pass': 0.3, 'band_stop': [0.3, 2], 'low_pass': 2}
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


def butter_filter(signal, highpass_freq=None, lowpass_freq=None, order=4, fs=2500):

    # The filter type is determined according to the values of cut-off frequencies
    Fn = fs / 2.
    if lowpass_freq and highpass_freq:
        if highpass_freq < lowpass_freq:
            Wn = (highpass_freq / Fn, lowpass_freq / Fn)
            btype = 'bandpass'
        else:
            Wn = (lowpass_freq / Fn, highpass_freq / Fn)
            btype = 'bandstop'
    elif lowpass_freq:
        Wn = lowpass_freq / Fn
        btype = 'lowpass'
    elif highpass_freq:
        Wn = highpass_freq / Fn
        btype = 'highpass'
    else:
        raise ValueError("Either highpass_freq or lowpass_freq must be given")

    # Filter signal
    b, a = butter(order, Wn, btype=btype, output='ba')
    if len(signal.shape) > 1:
        filtered_data = filtfilt(b=b, a=a, x=signal, axis=1)
    else:
        filtered_data = filtfilt(b=b, a=a, x=signal)

    return filtered_data


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


# Loop through sessions
results_df = pd.DataFrame()
for i, session_id in enumerate(sessions.index.values):
    print(f'Processing session {session_id} [{i+1} of {sessions.index.values.shape[0]}]')
    session = cache.get_session_data(session_id)

    # Load in crypto data from the time of the recording
    crypto_vector = get_crypto_vector(TICKER, session.session_start_time)

    # Get full region names
    these_units = session.units

    # Correlate units with unfiltered crypto trace
    for j, unit_id in enumerate(these_units.index.values):

        # Correlate neuron activity with crypto value
        activity_vector = get_activity_vector(session.spike_times[unit_id], BIN_SIZE)
        r, p = pearsonr(activity_vector, crypto_vector[:activity_vector.shape[0]])

        # Add to results dataframe
        results_df = results_df.append(pd.DataFrame(index=[results_df.shape[0] + 1], data={
            'r': r, 'p': p, 'session_id': session_id, 'unit_id': unit_id,
            'subject': session.specimen_name, 'filt': 'none', 'filt_freq': np.nan,
            'acronym': these_units.loc[unit_id]['ecephys_structure_acronym']}))

    # Loop over filters
    for k, filt in enumerate(FILT.keys()):

        # Filter crypto trace
        if filt == 'high_pass':
            crypto_filt = butter_filter(crypto_vector, highpass_freq=FILT[filt], fs=60)
        elif filt == 'band_stop':
            crypto_filt = butter_filter(crypto_vector, highpass_freq=FILT[filt][1],
                                        lowpass_freq=FILT[filt][0], fs=60)
        elif filt == 'low_pass':
            crypto_filt = butter_filter(crypto_vector, lowpass_freq=FILT[filt], fs=60)

        # Correlate units with filtered trace
        for j, unit_id in enumerate(these_units.index.values):

            # Correlate neuron activity with crypto value
            activity_vector = get_activity_vector(session.spike_times[unit_id], BIN_SIZE)
            r, p = pearsonr(activity_vector, crypto_filt[:activity_vector.shape[0]])

            # Add to results dataframe
            results_df = results_df.append(pd.DataFrame(index=[results_df.shape[0] + 1], data={
                'r': r, 'p': p, 'session_id': session_id, 'unit_id': unit_id,
                'subject': session.specimen_name, 'filt': filt, 'filt_freq': [FILT[filt]],
                'acronym': these_units.loc[unit_id]['ecephys_structure_acronym']}))

# Save results
results_df.to_csv(join(data_dir, f'{NAME}_correlations_filter.csv'))
