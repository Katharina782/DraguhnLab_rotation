#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 19:13:46 2021

@author: kmikulik
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#from datasheet_mod_v01 import datasheet_mod_v01
#mport pingouin as pg
#rom pingouin import pairwise_corr, read_dataset
sns.set(style="ticks", palette='rocket')
from matplotlib import rc
rc("pdf", fonttype=42)
#from sinaplot import  sinaplot
def cm2inch(value):
    return value/2.54
from scipy.signal import savgol_filter
import stfio
from scipy.signal import find_peaks
#from CT_functions_v01 import determine_threshold, determine_Ifreq, resample,function1d

import os
from os import listdir
import errno
import math
import re



file = "/mnt/live1/kmikulik/recordings/Analysis/2021_03_17/MKsinus_074.cfs"


def read_file(file_name):
    rec = stfio.read(file_name, "tcfs")
    # convert data into an array
    rec = np.array(rec)
    return rec

def create_timeline_array():
    # np.arrange creates arrays with regularly incrementing values
    # first value is start of array, second is end of array
    # in this case 0.02 will be the increment
    timeline = np.arange(0, 2000, 0.02).tolist()
    timeline = np.asarray(timeline)
    return (timeline)

rec = read_file(file)
timeline = create_timeline_array()

vol = rec[1][26]
cur = rec[0][26]

"""
First we do the correlation as an autocorrelation between one and the same current trace
"""
#when we perform the correlation we get 200 000 entries, because we correlate 100 000 data points with 100 000 datapoints
corr = np.correlate(cur, cur, "full")

#now we trie to only take the interval from the perfect correlation at 100 000 - 1000 until +1000 datapoints to the right
#we take 1000 datapoints, because that is one period

corr_interval = corr[int(np.round(len(corr)/2))-1000:int(np.round(len(corr)/2)+1000)]

#in this plot we can see that the best correlation is exactly at 1000, when there is a perfect fit of the two waves
#in this case there is a perfect fit at 1000, because it is an autocorrelation and there is no phase delay
#if there was a phase delay then we would see the best correlation(maximum correlation value) at some point to the right of 1000
plt.plot(corr_interval)



"""
Now we try the same autocorrelation with one and the same voltage trace
"""
#when we perform the correlation we get 200 000 entries, because we correlate 100 000 data points with 100 000 datapoints
corr = np.correlate(vol, vol, "full")

#now we trie to only take the interval from the perfect correlation at 100 000 - 1000 until +1000 datapoints to the right
#we take 1000 datapoints, because that is one period

corr_interval = corr[int(np.round(len(corr)/2))-1000:int(np.round(len(corr)/2)+1000)]

#in this plot we can see that the best correlation is exactly at 1000, when there is a perfect fit of the two waves
#in this case there is a perfect fit at 1000, because it is an autocorrelation and there is no phase delay
#if there was a phase delay then we would see the best correlation(maximum correlation value) at some point to the right of 1000
plt.plot(corr_interval)




"""
Now we try a correlation between voltage and current trace, whcih is not an autocorrelation anymore
"""
corr = np.correlate(cur, vol, "full")

#now we trie to only take the interval from the perfect correlation at 100 000 - 1000 until +1000 datapoints to the right
#we take 1000 datapoints, because that is one period

corr_interval = corr[int(np.round(len(corr)/2)):int(np.round(len(corr)/2)+1000)]

#in this plot we can see that the best correlation is exactly at 1000, when there is a perfect fit of the two waves
#in this case there is a perfect fit at 1000, because it is an autocorrelation and there is no phase delay
#if there was a phase delay then we would see the best correlation(maximum correlation value) at some point to the right of 1000
plt.plot(corr_interval)
#plt.savefig("/mnt/live1/kmikulik/plots/crosscorrelation_zoomed_in.png")

mv = np.min(corr_interval)

mi = np.where(corr_interval==mv)[0]

 

time_lag = mi*0.02

#The time lag we get from the correlation is 6.66 ms, but in the file it looks like 3 ms


#When we plot the results, shifting the voltage trace 333 datapoints to the left, it seems to shift the trace too far to the left

plt.plot(cur[0:1000])
plt.show()
plt.plot(vol[333:1333])




"""
take the first part of the voltage and current trace
the first 100ms
"""
ind = int(100/0.02)

voli= vol[0:ind]
curi= cur[0:ind]


corr = np.correlate(curi, voli, "full")

#now we trie to only take the interval from the perfect correlation at 100 000 - 1000 until +1000 datapoints to the right
#we take 1000 datapoints, because that is one period

corr_interval = corr[int(np.round(len(corr)/2)):int(np.round(len(corr)/2)+1000)]

#in this plot we can see that the best correlation is exactly at 1000, when there is a perfect fit of the two waves
#in this case there is a perfect fit at 1000, because it is an autocorrelation and there is no phase delay
#if there was a phase delay then we would see the best correlation(maximum correlation value) at some point to the right of 1000
plt.plot(corr_interval)
#plt.savefig("/mnt/live1/kmikulik/plots/crosscorrelation_zoomed_in.png")

mv = np.min(corr_interval)

mi = np.where(corr_interval==mv)[0]

 

time_lag = mi*0.02










"""
do the exact same thing for a different file
Here the maximal value for the crosscorrelation is shifted to the left, not to the right
However the voltage trace is still shifted to the right, not to the left...
In this case the value we get for the phase delay seems to be reasonable
"""

file = "/mnt/live1/kmikulik/recordings/Analysis/2021_03_17/MKsinus_069.cfs"


def read_file(file_name):
    rec = stfio.read(file_name, "tcfs")
    # convert data into an array
    rec = np.array(rec)
    return rec

def create_timeline_array():
    # np.arrange creates arrays with regularly incrementing values
    # first value is start of array, second is end of array
    # in this case 0.02 will be the increment
    timeline = np.arange(0, 2000, 0.02).tolist()
    timeline = np.asarray(timeline)
    return (timeline)

rec = read_file(file)
timeline = create_timeline_array()

vol = rec[1][25]
cur = rec[0][25]

"""
First we do the correlation as an autocorrelation between one and the same current trace
"""
#when we perform the correlation we get 200 000 entries, because we correlate 100 000 data points with 100 000 datapoints
corr = np.correlate(cur, cur, "full")

#now we trie to only take the interval from the perfect correlation at 100 000 - 1000 until +1000 datapoints to the right
#we take 1000 datapoints, because that is one period

corr_interval = corr[int(np.round(len(corr)/2))-1000:int(np.round(len(corr)/2)+1000)]

#in this plot we can see that the best correlation is exactly at 1000, when there is a perfect fit of the two waves
#in this case there is a perfect fit at 1000, because it is an autocorrelation and there is no phase delay
#if there was a phase delay then we would see the best correlation(maximum correlation value) at some point to the right of 1000
plt.plot(corr_interval)



"""
Now we try the same autocorrelation with one and the same voltage trace
"""
#when we perform the correlation we get 200 000 entries, because we correlate 100 000 data points with 100 000 datapoints
corr = np.correlate(vol, vol, "full")

#now we trie to only take the interval from the perfect correlation at 100 000 - 1000 until +1000 datapoints to the right
#we take 1000 datapoints, because that is one period

corr_interval = corr[int(np.round(len(corr)/2))-1000:int(np.round(len(corr)/2)+1000)]

#in this plot we can see that the best correlation is exactly at 1000, when there is a perfect fit of the two waves
#in this case there is a perfect fit at 1000, because it is an autocorrelation and there is no phase delay
#if there was a phase delay then we would see the best correlation(maximum correlation value) at some point to the right of 1000
plt.plot(corr_interval)




"""
Now we try a correlation between voltage and current trace, whcih is not an autocorrelation anymore
"""
corr = np.correlate(cur, vol, "full")

#now we trie to only take the interval from the perfect correlation at 100 000 - 1000 until +1000 datapoints to the right
#we take 1000 datapoints, because that is one period

corr_interval = corr[int(np.round(len(corr)/2)) - 1000:int(np.round(len(corr)/2)+1000)]

#in this plot we can see that the best correlation is exactly at 1000, when there is a perfect fit of the two waves
#in this case there is a perfect fit at 1000, because it is an autocorrelation and there is no phase delay
#if there was a phase delay then we would see the best correlation(maximum correlation value) at some point to the right of 1000
plt.plot(corr_interval)
#plt.savefig("/mnt/live1/kmikulik/plots/crosscorrelation_zoomed_in.png")

mv = np.max(corr_interval)

mi = np.where(corr_interval==mv)[0]

delayi = 1000-mi

delay_time = delayi * 0.02



plt.plot(cur[0:1000])
plt.plot(vol[171:1171])