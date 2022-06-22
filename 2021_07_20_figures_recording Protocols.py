#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 10:20:52 2021

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



#file = "/mnt/live1/kmikulik/recordings/Analysis/2021_03_17/MKsinus_069.cfs"
cellchar = "/mnt/live1/kmikulik/recordings/Analysis/2021_03_17/MKcellchar_119.cfs"
spulse = "/mnt/live1/kmikulik/recordings/Analysis/2021_03_17/MKspulse_305.cfs"
sinus = "/mnt/live1/kmikulik/recordings/Analysis/2021_03_17/MKsinus_074.cfs"

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

timeline = create_timeline_array()




""" 
plot sinus protocol for presentation
5hz
20Hz
"""

#5 hertz

rec_sinus = read_file(sinus)
vol = rec_sinus[1]
cur = rec_sinus[0]

fig, ax = plt.subplots(2)
fig.set_size_inches(20, 12)
for i in range (4, 8):
    #upper panel
    ax[0].plot(timeline, vol[i])
    ax[0].set_ylim( [-60, 70])
    ax[0].set_xlim( [0, 2000])
    x = np.arange(0, 2250, 250).tolist()
    y = np.arange(-60, 80, 20).tolist()
    ax[0].set_xticklabels(x, fontsize=25)
    ax[0].set_yticklabels(y, fontsize=25)
    ax[0].set_ylabel("cell membrane potential [mV]").set_size(30)
    
    #lower panel
    ax[1].plot(timeline, cur[i])
    ax[1].set_ylim ( [0, 100])
    ax[1].set_xlim ( [0, 2000])
    z = np.arange(-100, 200, 10).tolist()
    ax[1].set_xticklabels(x, fontsize=25)
    ax[1].set_yticklabels(z, fontsize=25)
    ax[1].set_xlabel ( "time [ms]" ).set_size(30)
    ax[1].set_ylabel ("current pulse [pA]").set_size(30)
plt.show
plt.savefig("/mnt/live1/kmikulik/plots/sinus_5Hz_figure.png")


#20 Herzt
#frame 13 - 19

fig, ax = plt.subplots(2)
fig.set_size_inches(20, 12)

for i in range (13, 19):
    #upper panel
    ax[0].plot(timeline, vol[i])
    ax[0].set_ylim( [-80, 70])
    ax[0].set_xlim( [0, 2000])
    x = np.arange(0, 2250, 250).tolist()
    y = np.arange(-80, 80, 20).tolist()
    ax[0].set_xticklabels(x, fontsize=25)
    ax[0].set_yticklabels(y, fontsize=25)
    ax[0].set_ylabel("cell membrane potential [mV]").set_size(30)
    
    #lower panel
    ax[1].plot(timeline, cur[i])
    ax[1].set_ylim ( [-50, 150])
    ax[1].set_xlim ( [0, 2000])
   # z = np.arange(-50, 150, 10).tolist()
    ax[1].set_xticklabels(x, fontsize=25)
    #ax[1].set_yticklabels(z, fontsize=25)
    ax[1].set_xlabel ( "time [ms]" ).set_size(30)
    ax[1].set_ylabel ("current pulse [pA]").set_size(30)
plt.show
plt.savefig("/mnt/live1/kmikulik/plots/sinus_20Hz_figure.png")



"""
plot spulse for the presentation
"""

spulse = "/mnt/live1/kmikulik/recordings/2021_03_10/MKspulse_129.cfs"

rec_spulse = read_file(spulse)
vol = rec_spulse[1]
cur = rec_spulse[0]

time = np.arange(0, 500, 0.02).tolist()

    

fig, ax = plt.subplots(2)
fig.set_size_inches(18.5, 15)
for i in range(0, len(vol)-1):
    #upper panel
    ax[0].plot(time, vol[i], color = "black")
    ax[0].set_ylim( [-80, 80] )
    ax[0].set_xlim( [140, 170])
    x = np.arange(140, 175, 5).tolist()
    y = np.arange(-80, 80, 20).tolist()
    ax[0].set_xticklabels(x, fontsize=25)
    ax[0].set_yticklabels(y, fontsize=25)
    ax[0].set_ylabel("cell membrane potential [mV]").set_size(30)
    ax[0].axvline(x = 160, color = "r")
    
    #lower panel
    ax[1].plot(time, cur [i], color = "black")
    ax[1].set_ylim ( [-120, 400])
    ax[1].set_xlim ( [140, 170])
    x = np.arange(140, 175, 5).tolist()
    z = np.arange(-100, 500, 100).tolist()
    ax[1].set_xticklabels(x, fontsize=25)
    ax[1].set_yticklabels(z, fontsize=25)
    ax[1].set_xlabel ( "time [ms]" ).set_size(30)
    ax[1].set_ylabel ("current pulse [pA]").set_size(30)
plt.show
plt.savefig("/mnt/live1/kmikulik/plots/spulse_figure.png")



"""
plot cellchar for the presentation
"""

rec_cellchar = read_file(cellchar)
vol = rec_cellchar[1]
cur = rec_cellchar[0]


fig01, ax1 = plt.subplots(2) # sharey = sharing of y axis beschriftung
fig01.set_size_inches(18.5, 15)

for i in range(0, (len(vol)-1)):
    #print(i)
    ax1[0].plot(timeline, vol[i], color = "black")
    ax1[0].set_ylim([-120, 50])
    ax1[0].set_xlim([700,1400])
    x = np.arange(700, 1500, 100).tolist()
    y = np.arange(-120, 60, 20).tolist()
    ax1[0].set_xticklabels(x, fontsize=25)
    ax1[0].set_yticklabels(y, fontsize=25)
    ax1[0].set_ylabel("cell membrane potential [mV]").set_size(30)
#axs[0].plot(x_time, np.zeros_like(y_voltage),'--',color='gray')
    ax1[1].plot(timeline, cur[i], color = "black") 
    ax1[1].set_ylim([-300, 500])
    ax1[1].set_xlim([700, 1400])
    z = np.arange(-300, 600, 100).tolist()
    ax1[1].set_xticklabels(x, fontsize=25)
    ax1[1].set_yticklabels(z, fontsize=25)
    ax1[1].set_xlabel("time [ms]").set_size(30)
    #ax1[1]. xaxis. label. set_size(20)
    ax1[1].set_ylabel("current pulse [pA]").set_size(30)
plt.show
plt.savefig("/mnt/live1/kmikulik/plots/cellchar_figure.png")
    
