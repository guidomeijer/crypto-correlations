#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 17:43:08 2021

@author: Guido Meijer
"""

import random
import time
import datetime
import numpy as np
from os import mkdir
from os.path import join, dirname, exists, realpath
from Historic_Crypto import HistoricalData

# Settings
TICKER = 'BTC'
ITERATIONS = 1000
BIN_SIZE = 60  # seconds
DURATION = 5  # hours

# Set directory
data_dir = join(dirname(realpath(__file__)), 'data')
if not exists(data_dir):
    mkdir(data_dir)


def str_time_prop(start, end, format, prop):
    """Get a time at a proportion of a range of two formatted times.

    start and end should be strings specifying times formated in the
    given format (strftime-style), giving an interval [start, end].
    prop specifies how a proportion of the interval to be taken after
    start.  The returned time will be in the specified format.
    """

    stime = time.mktime(time.strptime(start, format))
    etime = time.mktime(time.strptime(end, format))

    ptime = stime + prop * (etime - stime)

    return time.strftime(format, time.localtime(ptime))


def random_date(start, end, prop):
    return str_time_prop(start, end, '%Y-%m-%d-%H-%M', prop)


# Get a null distribution of crypto vectors
null_crypto = []
while len(null_crypto) < ITERATIONS:
    start_time = random_date('2019-01-01-12-00', '2020-01-01-12-00', random.random())
    end_time = datetime.datetime.strptime(start_time, '%Y-%m-%d-%H-%M') + datetime.timedelta(hours=5)
    try:
        crypto_data = HistoricalData(f'{TICKER}-USD', BIN_SIZE, start_time,
                                     end_time.strftime("%Y-%m-%d-%H-%M")).retrieve_data()
        if crypto_data['open'].values.shape[0] == (DURATION * 60 * 60) / BIN_SIZE:
            null_crypto.append(crypto_data['open'].values)
    except:
        print('Error while fetching crypto data')

# Convert to ndarray
null_crypto = np.array(null_crypto)

# Save to disk
np.save(join(data_dir, f'{TICKER}_null_vectors.npy'), null_crypto)

