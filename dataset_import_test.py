# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 16:17:58 2020

@author: Dell
"""
# %% importing dependancies
from pyedflib import EdfReader
import numpy as np

# %% importing the dataset
filepath = 'D:\Mini II\DATASETS\eNTERFACING\Data\EEG\Part1_IAPS_SES1_EEG_fNIRS_03082006.bdf'
f = EdfReader(filepath)
n = f.signals_in_file
signal_labels = f.getSignalLabels()
sigbufs = np.zeros((n, f.getNSamples()[0]))
for i in np.arange(n):
    sigbufs = f.readSignal(i)

# %% 