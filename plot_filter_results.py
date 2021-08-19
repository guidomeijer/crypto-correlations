#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 15:49:38 2021
By: Guido Meijer
"""

import numpy as np
import datetime
import seaborn as sns
from scipy.signal import filtfilt, butter
import matplotlib.pyplot as plt
from Historic_Crypto import HistoricalData

BIN_SIZE = 60  # seconds
TICKER = 'ETH'
NAME = 'Ethereum'
FILT = {'high_pass': 0.1, 'band_stop': [0.3, 2], 'low_pass': 2}
CRYPTO_TIME = datetime.datetime.strptime('2019-01-19-00-54', "%Y-%m-%d-%H-%M")  # some random time


def butter_filter(signal, highpass_freq=None, lowpass_freq=None, order=4, fs=60):

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


def get_crypto_vector(ticker, start_time):
    end_time = start_time + datetime.timedelta(hours=5)
    crypto_data = HistoricalData(f'{TICKER}-USD', BIN_SIZE,
                                 start_time.strftime("%Y-%m-%d-%H-%M"),
                                 end_time.strftime("%Y-%m-%d-%H-%M")).retrieve_data()
    return crypto_data['open'].values


# Get example crypto vector
crypto_vector = get_crypto_vector(TICKER, CRYPTO_TIME)
time_vector = np.linspace(0, crypto_vector.shape[0]*BIN_SIZE, crypto_vector.shape[0]) / 60

sns.set_context('talk')
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5), dpi=150)

ax1.plot(time_vector, crypto_vector, color=sns.color_palette('colorblind')[3], label='Original')
ax1.set(ylabel='Value of Ethereum (USD)', xlabel='Time (s)')

crypto_filt = butter_filter(crypto_vector, highpass_freq=0.3)
crypto_filt = (crypto_filt - np.min(crypto_filt)) / (np.max(crypto_filt) - np.min(crypto_filt))
ax2.plot(time_vector, crypto_filt, label='High pass', color=sns.color_palette('Set2')[0])

crypto_filt = butter_filter(crypto_vector, highpass_freq=2, lowpass_freq=0.3)
crypto_filt = (crypto_filt - np.min(crypto_filt)) / (np.max(crypto_filt) - np.min(crypto_filt))
ax2.plot(time_vector, crypto_filt, label='Band stop', color=sns.color_palette('Set2')[1])

crypto_filt = butter_filter(crypto_vector, lowpass_freq=2)
crypto_filt = (crypto_filt - np.min(crypto_filt)) / (np.max(crypto_filt) - np.min(crypto_filt))
ax2.plot(time_vector, crypto_filt, label='Low pass', color=sns.color_palette('Set2')[2])

ax2.legend(loc='center left', bbox_to_anchor=(1, 0.8), frameon=False)
ax2.set(ylabel='Normalized Ethereum value', xlabel='Time (s)')

plt.tight_layout()
sns.despine(trim=True)

