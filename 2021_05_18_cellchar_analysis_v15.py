#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 12:31:43 2021

@author: kmikulik
"""

# %%import libraries
# Import libraries.
import stfio
from xlwt import Workbook
import xlwt
from os import listdir
import errno
import os
import math
from scipy.signal import find_peaks
import re
from matplotlib import rc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set(style="ticks", palette='rocket')

rc("pdf", fonttype=42)

#from sinaplot import  sinaplot


def cm2inch(value):
    return value/2.54



# %%
# Testfile
#efile1 = "mnt/live1/cthome/CThome_ProjectArchive/Y_AISdiversity_PYR/MikulikKatharina_Data/KM_Data/23_02_2021"
#efile1 = "/mnt/live1/kmikulik/recordings/23_02_2021/MKcellchar_021.cfs"
# #efile1 = "/mnt/live1/jwinterstein/LayerVI_EC/2021/04_2021/130421/Slice1/cellchar_432.cfs"
efile1 = "/mnt/live1/kmikulik/recordings/Analysis/2021_03_03/MKcellchar_629.cfs"
#efile1 = "/mnt/live1/kmikulik/recordings/2021_03_20/MKcellchar_125.cfs"
# array_3d = read_as_3Darray(efile1)
# timeline  = create_timeline_array()
# variables = set_variables(array_3d)
# analyse(frame_number)


# %%

def read_as_3Darray(file_name):
    rec = stfio.read(file_name, "tcfs")
    # convert data into an array
    array_3d = np.array(rec)
    # np.arrange creates arrays with regularly incrementing values
    # first value is start of array, second is end of array
    # in this case 0.02 will be the increment
    return array_3d


def create_timeline_array():
    timeline = np.arange(0, 2000, 0.02).tolist()
    timeline = np.asarray(timeline)
    return (timeline)


# %% set_parameters

def voltage_channel(array_3d):
    vol = array_3d[1]
    return vol


def current_channel(array_3d):
    cur = array_3d[0]
    return cur


def first_voltage_frame(array_3d):
    vol = array_3d[1]
    vol0 = vol[0]
    return vol0


def first_current_frame(array_3d):
    cur = array_3d[0]
    cur0 = cur[0]
    return cur0


def number_of_frames(array_3d):
    vol = array_3d[1]
    frame_number = len(vol)
    return frame_number


# def time_intervals():
#     #convert time to data points
#     global time250, time305, time395
#     time250 = int(250/0.02)
#     time305 = int(305/0.02)
#     time395 = int(395/0.02)

# #def set_variables(array_3d):
#     global vol, cur, vol0, cur0, frame_number
#     vol = voltage_channel(array_3d)
#     cur = current_channel(array_3d)
#     vol0 = first_voltage_frame(array_3d)
#     cur0 = first_current_frame(array_3d)
#     frame_number = number_of_frames(array_3d)
#     time_intervals()


# %%
"""
Holding current, resting membrane potential of just the first frame
"""


def holding_current_rmp(timeline, array_3d):
    cur0 = first_current_frame(array_3d)
    vol0 = first_voltage_frame(array_3d)
    # we define the time interval where we can calculate the rmp and holding current
    interval = np.where((timeline > 0) & (timeline < 250))
    # we calculate rmp and holding current for the first frame
    rmp = np.mean(vol0[interval])
    hold = np.mean(cur0[interval])
    #print("resting membrane potential", rmp, "holding current", hold)
    return rmp, hold


"""
Calculate input resistance of just the first frame 
"""


def input_resistance(timeline, array_3d):
    cur0 = first_current_frame(array_3d)
    vol0 = first_voltage_frame(array_3d)
    # we calculate the holding curret and rmp
    rmp, hold = holding_current_rmp(timeline, array_3d)
    # we define the time interval where we can calculate the rmp and holding current
    interval = np.where((timeline > 0) & (timeline < 250))
    # now we define the time interval where we inject our test current pulse of -25 pA
    test_interval = np.where((timeline > 305) & (timeline < 395))
    # get the average current and voltage of this interval
    testcur = np.mean(cur0[test_interval])
    testvol = np.mean(vol0[test_interval])
    # get the difference between holding current / rmp and current and voltage at testpulse
    # since the testcur will always be negative we add the holding current, which can be poitive
    # or negative to get the absolute difference
    deltacur = abs(testcur - hold)
    # since the testvol will always be more negative than the rmp7, we can subtract the absolute values to get the difference
    deltavol = abs(testvol) - abs(rmp)
    # to calculate the input resistance we use
    #R = U [mV] / I [pA]
    # units Megaohm
    input_res = (deltavol / deltacur) * 1000
    return input_res, testcur, testvol, deltavol, deltacur
    #print("input resistance", input_res, "delta U", deltavol, "delta I", deltacur)


# %% Ih sag
"""
Hyperpolarization-activated current (Ih)
Calculate the difference between the hyperpolarization at the beginning of the 500ms current injection and the stable hyperpolarization
"""


def Ih_sag(timeline, array_3d):
    cur0 = first_current_frame(array_3d)
    vol0 = first_voltage_frame(array_3d)
    rmp, hold = holding_current_rmp(timeline, array_3d)
    # define the time interval where the 500ms current pulse is injected
    time_hyp = np.where((timeline > 800) & (timeline < 1300))
    # define timeinterval where the hyperpolarization is assumed to be stable
    time_stabhyp = np.where((timeline > 1100) & (timeline < 1300))
    # curinj = np.mean(cur0[time_hyp]) - hold_av #strength of current injection
    minV = np.min(vol0[time_hyp]) - rmp  # maximum hyperopolarization for Ih
    # stable hyperpolarization for Ih
    stabV = np.mean(vol0[time_stabhyp]) - rmp
    Ih_vol = minV/stabV
    #print(Ih_vol, minV, stabV)
    return minV, stabV, Ih_vol


# %% number of AP per frame and figures
"""
Find all peak (action potentials) in each frame.
Create a list of indices for all peaks -> peak_list
Create a list of the number of action potentials per frame -> AP_list
Create a list of increasing steps of current injections -> cur_list 
Create a list of the total number of peaks -> totalAP_list
"""


def AP_per_frame(timeline, array_3d):
    cur = current_channel(array_3d)
    vol = voltage_channel(array_3d)
    frame_number = number_of_frames(array_3d)
    # set pA to -225 so we can add +25 for each iteration through the loop
    pA = -225
    # create empty lists, for counting later on
    frame_list = list()
    cur_list = list()
    AP_list = list()
    peak_list = list()
    totalAP_list = list()
    total = 0
    # we iterate through all files of that frame
    for i in range(0, frame_number):
        # find peaks returns ndarray -> indices of peaks that satisfy all given conditions
        peaks, _ = find_peaks(vol[i], height=0)
        # set count to zero before iterating through the array with all peaks and adding 1 to count for each peak
        count = 0
        # by adding +25 for each iteration through frames we know the current injection of that frame
        pA = pA + 25
        for peak in peaks:
            peak_list.append(peak)
            # for each peak we add  1 to count. For every new frame count is set to 0 again -> gives us number of counts per frame
            count = count + 1
            # total is not reset to 0 for every frame -> gives us the total number of AP in all frames
            total = total + 1
        frame_list.append(i)
        AP_list.append(count)
        cur_list.append(pA)
        totalAP_list.append(total)
        #print(i, count, pA)
    return cur_list, frame_list, AP_list, peak_list, totalAP_list


"""
Input-Output curve:
Plot the number of action potentials versus current injection.
"""


def plot_input_output(timeline, date, file, array_3d):
    cur_list, frame_list, AP_list, peak_list, totalAP_list = AP_per_frame(
        timeline, array_3d)
    # create a second figure for the input_output curve and save the figure as a pdf in the specified directory
    fig02, ax2 = plt.subplots()
    ax2.plot(cur_list, AP_list, "bo", color="red", markersize=3)
    ax2.set_title("input - output")
    ax2.set_xlabel("injected current [pA]")
    ax2.set_ylabel("number of action potentials")
    pdf_name = f"/mnt/live1/kmikulik/recordings/Analysis/{date}/{file}_input_output_curve.png"
    #pp = PdfPages(pdf_name)
    # pp.savefig(fig01)
    plt.savefig(pdf_name)


"""
We want to plot voltage and current versus time for each frame
"""


#def plot_frames(timeline, array_3d):
    # cur = current_channel(array_3d)
    # vol = voltage_channel(array_3d)
    # frame_number = number_of_frames(array_3d)
    # # we iterate through all files of that frame
    # for i in range(0, frame_number):
    #     # for each frame we create a figure plotting current and voltage versus time
    #     fig01, ax1 = plt.subplots(2)
    #     fig01.suptitle(i)
    #     ax1[0].plot(timeline, vol[i])
    #     ax1[0].set_ylim([-120, 50])
    #     ax1[0].set_xlabel("time [ms]")
    #     ax1[0].set_ylabel("voltage [mV]")
    #     ax1[1].plot(timeline, cur[i])
    #     ax1[1].set_ylim([-200, 500])
    #     ax1[1].set_ylabel("current [pA]")
    #     # find peaks returns ndarray -> indices of peaks that satisfy all given conditions
    #     peaks, _ = find_peaks(vol[i], height=0)
    #     for peak in peaks:
    #         ax1[0].axvline(x=timeline[peak])


# %% rheobase
"""
Rheobase is the current you have to inject to trigger at least one action potential
"""


def rheobase(timeline, array_3d):
    cur_list, frame_list, AP_list, peak_list, totalAP_list = AP_per_frame(
        timeline, array_3d)
    for i in AP_list:
        if i < 1:
            continue
        # the first time we have an action potential in the list of APs we get the index of that AP
        # with this index we can get the injected current from the list o fcurrent injections
        if i >= 1:
            pos = AP_list.index(i)
            rheo = cur_list[pos]
            print("Rheobase:", rheo)
            break
    return rheo


# %%
""" 
ISI burstratio
ratio between timeintervals of first two and last two action potentials of a 500ms current injection
to calculate this we want to use a current injection where there are at least 7 action potentials
if in a file we do not find a current injection that triggers at least 7 action potentials we set burstratio and timeintervals to zero
"""


def ISI(timeline, array_3d):
    vol = voltage_channel(array_3d)
    cur_list, frame_list, AP_list, peak_list, totalAP_list = AP_per_frame(
        timeline, array_3d)
    # take the frame that has at least 7 spikes
    for i in AP_list:
        if i < 7:
            frame = "no"
            continue
        if i >= 7:
            frame = AP_list.index(i)
            break
    # now we know with which frame to work
    # to find positions of peaks 1, 2, 6, 7 we have to determine the index of the peaks
    # number of total AP is one bigger than index of list p
    if frame == "no":
        #this is for safety. In case we encounter a  file that has not enough action potentials we set the burstratio to zero
        burstratio = 0
        interval1 = 0
        interval2 = 0
    else:
        # create a 1D array with only the voltage of the first frame that has at least 7 spikes
        # find peaks in that frame
        # calculate the interval between peak 1 and 2 and peak 6 and 7
        # multiply by 0.02 to get the timeinterval in ms
        vol_isi = vol[frame]
        peaks, _ = find_peaks(vol_isi, height=0)
        interval1 = ((peaks[1]) - (peaks[0])) * 0.02
        interval2 = ((peaks[6]) - (peaks[5])) * 0.02
        burstratio = interval2 / interval1
        #print("burstratio", burstratio)
    return burstratio, interval1, interval2


# %% upsampling
def nyquist_upsampling(data):

    import pandas as pd

    dummy = pd.read_csv('/mnt/live1/mboth/PythonWS/Interpolation_kernel.csv')
    kernel = dummy.to_numpy()
#    d_u = 0

    up_fac = 10
#    dum = np.sin( np.arange(1,100,1) ) # the originatl data
    dum = data

    num_zeros = up_fac-1  # upFac-1
    dum_u = np.reshape(np.concatenate((np.array([dum]), np.zeros(
        [num_zeros, len(dum)]))).T, (1, (num_zeros+1)*len(dum)))[0]  # upsample in matlab

    nm = len(kernel)
    nx = len(dum_u)
    if nm % 2 != 1:    # n even
        m = nm/2
    else:
        m = (nm-1)/2

    X = np.concatenate(
        (np.ones(int(m))*dum_u[0], dum_u, np.ones(int(m))*dum_u[-1]))
    y = np.zeros(nx)

    indc = np.repeat(np.expand_dims(np.arange(0, nx, 1), axis=1), nm, axis=1).T
    indr = np.repeat(np.expand_dims(np.arange(0, nm, 1), axis=1), nx, axis=1)
    ind = indc + indr

    # xx = X(ind) this is what you do in matlab: use the array of indices to construct a new array with the values at the indices
    xx = np.zeros(ind.shape)
    for cX in range(0, nm):
        for cY in range(0, nx):
            xx[cX, cY] = X[ind[cX][cY]]

    y = np.inner(kernel.T, xx.T)[0].T

    return y
# %%


def AP_upsample(x_time, y_voltage):
    y_voltage_ny = nyquist_upsampling(y_voltage)
    x_time_ny = np.linspace(x_time[0], x_time[-1], len(y_voltage_ny))

    y_voltage_ny = y_voltage_ny[99:-100:1]
    x_time_ny = x_time_ny[99:-100:1]

    return [x_time_ny, y_voltage_ny]


# %%
"""
Find the first AP of a recording with a threshold within the first 10 ms
This will be the AP to analyse -> ap_for_analysis
It will also give you the frame of this AP
This will also give you the threshold of this AP
also it will give you the intervals around this peak we want to analyse as a representative AP:
    y_voltage = vol_int and x_time = time_int for other functions of AP analysis
In case there is no AP within 10ms find the AP that has the smallest distance from the onset of the 500ms current injection
"""


def find_ap_10ms(timeline, array_3d):
    #cur_list, frame_list, AP_list, peak_list, totalAP_list = AP_per_frame( timeline )
    # to detrmine if the AP threshold is inside the first 10ms
    vol = voltage_channel(array_3d)
    peakpos = {}
    AP_list = []
    distance_dict = {}
    frame_list = np.arange(0, len(vol), 1).tolist()
    for frame in frame_list:
        peaks, peakind = find_peaks(vol[frame], height=0)
        # values for each frame are a list of indices where peaks are located in this frame
        peakpos[frame] = peaks
        count = 0
        for peak in peaks:
            count += 1
        AP_list.append(count)
    # define voltage and time interval of the first AP for each frame
    # for this we need to determine the threshold for all APs!
    first_ap_vol = {}
    first_ap_time = {}
    for frame in frame_list:
        if AP_list[frame] > 0:
            # we get voltage values around the first peaks (-100 to +150 data points from peak)
            # the first peak is found at the zeroeth position of the entries for a certain frame in peakpos dictionary
            # we create dictionary with frame as key and voltage or time intervals around the first peak in that frame as the value
            # we can access the first peak via our dictionary frames
            vol_first_peak = vol[frame][(
                peakpos[frame][0]-100):(peakpos[frame][0]+150)]
            time_first_peak = timeline[(
                peakpos[frame][0]-100):(peakpos[frame][0]+150)]
            first_ap_vol[frame] = vol_first_peak
            first_ap_time[frame] = time_first_peak
        else:
            continue
    # determine AP threshold for the first AP in each frame
    for frame in frame_list:
        if AP_list[frame] > 0:
            # set parameters for the upsampling function
            y_voltage = first_ap_vol[frame]
            x_time = first_ap_time[frame]
            thr_section = np.nan
            # use upsampling function defined above
            x_time_ny, y_voltage_ny = AP_upsample(x_time, y_voltage)
            # first derivative of voltage values in that interval
            dat_ds = np.diff(y_voltage_ny)/np.median(np.diff(x_time_ny))
            # indices where voltage slope > 20
            ind_thr = np.nonzero(dat_ds > 20)[0]
    # this gives you the threshold of this AP at the same time
            if len(ind_thr) > 0:
                thr_vol = y_voltage_ny[ind_thr[0]-1]
                ind_thr = ind_thr[0]-1
                thr_time = x_time_ny[ind_thr]
                #calculate the distance between onset of current injection at 800ms and the time of the AP threshold
                distance = thr_time  - 800
                distance_dict [frame] = distance

            if thr_time < 810:
                # this gives us the position of the peak that we want to analyse
                AP_distance = min(distance_dict.items(), key = lambda x : x[1]) [1] 
                ap_for_analysis = peakpos[frame][0]
                frame_ap_analysis = frame
                y_vol = vol[frame][(peakpos[frame][0]-100):(peakpos[frame][0]+150)]
                x_tim = timeline[(peakpos[frame][0]-100):(peakpos[frame][0]+150)]
                #print(ap_for_analysis, frame_ap_analysis, thr_section, y_vol, x_tim)
                break

            #if there was no thr below 810 ms we did not break the loop, so we 
            AP_distance = min(distance_dict.items(), key = lambda x : x[1]) [1] #just get the value of the distance, not the key frame
            frame_ap_analysis = min(distance_dict.items(), key = lambda x : x[1]) [0] # just the key and not the distance value
            ap_for_analysis = peakpos[frame][0]
            y_vol = vol[frame][(peakpos[frame][0]-100):(peakpos[frame][0]+150)]
            x_tim = timeline[(peakpos[frame][0]-100):(peakpos[frame][0]+150)]
                    
    return ap_for_analysis, AP_distance, frame_ap_analysis, thr_vol, y_vol, x_tim


# %%
"""
determine the onset rapidness of the AP.
Plot the onset rapidness -> voltage (mV) vs mV/ms
Save the figure in the specified directory
"""


def determine_rapidness(x_tim, y_vol, date, file, thr_vol):
    slope = np.nan
    #x_time, y_voltage = AP_upsample(x_time, y_voltage)
    #dat_ds = slope (mv/mS)
    dat_ds = np.diff(y_vol)/np.median(np.diff(x_tim))

    indrap = []
    for i in range(0, len(dat_ds)):
        if dat_ds[i] >= 15:
            indrap.append(i)
        if dat_ds[i] >= 50:
            break
    # indrap is a list of indices of those slope values of the action potential curve that are between 15 and 50
    # polyfit fits a polynomial to points x (y_voltage), y (dat_ds)
    # last number = degree 1 -> linear polynomial
    # y_voltage[indrap] creates a list of the voltage values where slope is between 15 and 50
    # dat_ds[indrap] creates a list of the slope values at the before determined indices indrap
    # the list of values for voltage and slope are fitted to a linear polynomial
    # polyfit returns the coefficient(slope of the linear function) and the y-intercept
    slope, intercept = np.polyfit(y_vol[indrap], dat_ds[indrap], 1)
    # this  gives us the slope and y intercept of a linear function
    # determines voltage range for plotting rapidness -> 3 voltage values around the threshold voltage
    thresllist = [thr_vol - 2,
                  thr_vol,
                  thr_vol + 5]
    # list of 3 voltage values
    APrapidness = [slope, intercept]
    # abline values = 3
    abline_values = [APrapidness[0] * i + APrapidness[1] for i in thresllist]

    # sharey = sharing of y axis beschriftung
    fig02, axs = plt.subplots(2, 1, figsize=(6, 6), sharex=False)
    fig02.suptitle('Action potential')

    axs[0].plot(x_tim, y_vol)
    axs[0].axhline(thr_vol, color='r', linestyle='-')
    axs[0].set_xlabel('time (ms)')
    axs[0].set_ylabel('voltage (mV)')
    # [0:-1], geht bis zum vorletzten Wert
    axs[1].plot(y_vol[0:-1], dat_ds)
    axs[1].axhline(y=20, color='r', linestyle='-')
    axs[1].plot(thresllist, abline_values, 'b')
    axs[1].set_xlabel('voltage (mV)')
    axs[1].set_ylabel('slope (mV/ms)')
    #print("rapidness", slope, intercept)
    # save the figure as a pdf in the specified directory
    pdf_name = f"/mnt/live1/kmikulik/recordings/Analysis/{date}/{file}_AP_rapidness.png"
    #pp = PdfPages(pdf_name)
    # pp.savefig(fig01)
    plt.savefig(pdf_name)
    return slope, intercept


# %% amplitude
def determine_amplitude(x_tim, y_vol, thr_vol):
    """ determine the amplitude of the AP. """
    APamp = np.nan
    #x_time, y_voltage = AP_upsample(x_time, y_voltage)
    low = thr_vol
    peakind, peaks = find_peaks(y_vol, height=0)
    # das zieht nur den Wert des peaks aus der Liste, weil die Liste nur einen value hat -> ersten index [0]
    up = peaks['peak_heights'][0]
    APamp = up-low
    #print("amp", APamp)
    return APamp


# %%halfwidth
def determine_halfwidth(x_tim, y_vol, thr_vol, APamp):
    """ determine the halfwidth of the AP. """
    APhw = np.nan
    # if we divide the amplitude by 2, we have half the distance between threshold and peak.
    # if we add this distance to the threshold, we get the voltage value at this half height
    hAPamp = thr_vol + (APamp/2)
    # np.where will print an array containing a list of all indices at which y-voltage satisfies the condition
    # with the first [0] we get the list at the first index of the array
    # with the second [0] we get the first index value of the list -> this is the point on the left flank of the cure
    firsthind = np.where(y_vol >= hAPamp)[0][0]
    # by taking the last value of the list, we get the point on the right flank of the curve
    lasthind = np.where(y_vol >= hAPamp)[0][-1]
    # now we subtract the time values at the left flank point and the right flank point in order to get the halfwidth
    APhw = x_tim[lasthind]-x_tim[firsthind]
    #print("halfwidth", APhw)
    return APhw


# %% AP rise time
def determine_APriseT(x_tim, y_vol, thr_vol, APamp):
    """ determine the halfwidth of the AP. """
    APriseT = np.nan
    #x_time, y_voltage = AP_upsample(x_time, y_voltage)
    AP10perc = thr_vol + (0.1*APamp)
    AP90perc = thr_vol + (0.9*APamp)
    firsthind = np.where(y_vol >= AP10perc)[0][0]
    lasthind = np.where(y_vol >= AP90perc)[0][0]
    APriseT = x_tim[lasthind]-x_tim[firsthind]
    #print("APrisetime", APriseT)
    return APriseT

# %% AP fall time


def determine_APfallT(x_tim, y_vol, thr_vol, APamp):
    """ determine the halfwidth of the AP. """
    APfallT = np.nan
    #x_time, y_voltage = AP_upsample(x_time, y_voltage)
    AP10perc = thr_vol + (0.1*APamp)
    AP90perc = thr_vol + (0.9*APamp)
    firsthind = np.where(y_vol >= AP90perc)[0][-1]
    lasthind = np.where(y_vol >= AP10perc)[0][-1]
    APfallT = x_tim[lasthind]-x_tim[firsthind]
    #print("APfalltiem", APfallT)
    return APfallT

# %% max AP  rise


def determine_APmaxrise(x_tim, y_vol):
    """ determine the halfwidth of the AP. """
    APmaxrise = np.nan
    #x_time, y_voltage = AP_upsample(x_time, y_voltage)
    dat_ds = np.diff(y_vol)/np.median(np.diff(x_tim))
    APmaxrise = dat_ds.max()
    #print("max AP rise", APmaxrise)
    return APmaxrise


 # %%
"""
all functions for AP analysis in one 
"""


def ap_analysis(date, file, timeline, array_3d):
    ap_for_analysis, AP_distance, frame_ap_analysis, thr_vol, y_vol, x_tim = find_ap_10ms(timeline, array_3d)
    #find_ap_10ms (vol, timeline)
    slope, intercept = determine_rapidness(x_tim, y_vol, date, file, thr_vol)
    APamp = determine_amplitude(x_tim, y_vol, thr_vol)
    APhw = determine_halfwidth(x_tim, y_vol, thr_vol, APamp)
    APriseT = determine_APriseT(x_tim, y_vol, thr_vol, APamp)
    APfallT = determine_APfallT(x_tim, y_vol, thr_vol, APamp)
    APmaxrise = determine_APmaxrise(x_tim, y_vol)
    return thr_vol, AP_distance, slope, intercept, APamp, APhw, APriseT, APfallT, APmaxrise


# %%
"""
decide whether to analyze
only if there are more than 20 frames the recording is worth analyzing
"""


def analyse_cellchar(date, file, timeline, array_3d):
    frame_number = number_of_frames(array_3d)
    if frame_number >= 20:
        rmp, hold = holding_current_rmp(timeline, array_3d)
        input_res, testcur, testvol, deltavol, deltacur = input_resistance(
            timeline, array_3d)
        minV, stabV, Ih_vol = Ih_sag(timeline, array_3d)
        cur_list, frame_list, AP_list, peak_list, totalAP_list = AP_per_frame(
            timeline, array_3d)
        #plot_frames(timeline, array_3d)
        plot_input_output(timeline, date, file, array_3d)
        rheo = rheobase(timeline, array_3d)
        burstratio, interval1, interval2 = ISI(timeline, array_3d)
        thr_vol, AP_distance, slope, intercept, APamp, APhw, APriseT, APfallT, APmaxrise = ap_analysis(
            date, file, timeline, array_3d)
    return (rmp, hold, input_res, testcur, testvol, deltavol, deltacur, minV, stabV, Ih_vol, rheo, burstratio,
            interval1, interval2, thr_vol, AP_distance, slope, intercept, APamp, APhw, APriseT, APfallT, APmaxrise, AP_list)


# %%
# Testfile
#efile1 = "mnt/live1/cthome/CThome_ProjectArchive/Y_AISdiversity_PYR/MikulikKatharina_Data/KM_Data/23_02_2021"
#efile1 = "/mnt/live1/kmikulik/recordings/23_02_2021/MKcellchar_021.cfs"
# #efile1 = "/mnt/live1/jwinterstein/LayerVI_EC/2021/04_2021/130421/Slice1/cellchar_432.cfs"
# efile1 = "/mnt/live1/kmikulik/recordings/2021_03_06/MKcellchar_017.cfs"
# #efile1 = "/mnt/live1/kmikulik/recordings/23_02_2021/MKcellchar021_cfs"
# array_3d = read_as_3Darray(efile1)
# timeline  = create_timeline_array()
# variables = set_variables(array_3d)
# analyse(frame_number)


# %%

# %%  MAIN FUNCTION
#

""" 
In an excel sheet we have all the cellchar files we want to analyse for a certain cell.
In our folders we have many additional files, some of which we do not need to analyse.
Using the excelsheet we get a list of all filenames we want to analyse.
"""

def files_for_analysis(cellsheet):
    # save excelsheet as datafram
    cellsheet_df = pd.read_excel(cellsheet)
    # get the column that contains the cellchar number
    cellchar_70mV = cellsheet_df.MKcellchar_70mV
    # convert entries to list
    cellchar_for_analysis = cellchar_70mV[15:284].tolist()
    excel_filename_list = list()
    for i in cellchar_for_analysis:
        # because the values in the list are floats we first convert them to a string and then split the decimal points
        i = str(i)
        i = str.split(i, ".")[0]
        try:
            # because some entries in the excel sheet do not have 3 digits, we convert all entries to 3 digit entries
            # this way we will have correct filenames of the format: MKcellchar_007.cfs
            i = int(i)
            i = "{:0>3d}".format(i)
        except:
            pass
        #create filenames with respective number of cellchar i 
        excel_filename = f"MKcellchar_{i}.cfs"
        excel_filename_list.append(excel_filename)
    return excel_filename_list


"""
We want to create a unique id for each cell that we recorded and stained, so we can combine recording data and histology later on.
The unique id will contain Slice number and Cell number "SxCy"
In the main function we will also add the date "YYYY_MM_DD_SxCy"
"""
def create_cell_id(cellsheet):
    # save excelsheet as datafram
    cellsheet_df = pd.read_excel(cellsheet)
    # get the columsn that contain the files for analysis
    cellchar_70mV = cellsheet_df.MKcellchar_70mV[15:284].tolist()
    # get the columns that contain Slice and Cell number
    slice_num = cellsheet_df.Slice[15:284].tolist()
    cell_num = cellsheet_df.Cell[15:284].tolist()
    # create empty list for cell id
    cell_dict = {}
    # iterate through both lists and get the slice and cell number, but only if the entry at index i of the cellchar_70mV list is not "nan"
    for i in range(len(cellchar_70mV)):
        no_entry = np.isnan(cellchar_70mV)

        if no_entry[i] == True:
            continue

        else:
            file_num = int( cellchar_70mV [i])
            cell = str(cell_num[i])
            sli = str(slice_num[i])
            cell_id = f"{sli}{cell}"
            cell_dict[file_num] = cell_id
    return cell_dict



"""
create a unique identity for each cell with slice number, cell number and date
"""
def get_cell_id_for_file (file, date, cell_dict):
    file_num = file.replace("MKspulse_", "")
    file_num = int( file_num.replace(".cfs", "") )
    day = date.replace("_", "")
    cell_id = f"{day}{cell_dict[file_num]}"
    return cell_id, file_num




"""
This function creates a dictionary called dates that contains all files in a folder with certain date
"""
def file_dictionary(directory):
    # listdir creates a list of all files in that directory
    directory_files = listdir(directory)
    # create a dictioniary with:
    #keys = dates
    # values = files in folder with that date
    dates = {}
    for date in directory_files:
        dates[date] = listdir(f"{directory}{date}")
    return dates



def filter_filename(dates, excel_filename_list):
    """
    filter for file with correct name: MKcellchar_xxx.cfs
    then loop through the dictionary and check if the filename in the directory is also included in the excel sheet list
    of files for analysis
    then create a list of files of interest (files_oi) to analyse
    """
    for date, files in dates.items():
        files_oi = []
        # loop through all files of each date
        for file in files:
            if re.search("cellchar.*.cfs", file) != None:
                # for every file of this date create a variable x
                # then check if x is in the list you created from the excel sheet
                # if the file is on that list, you want to analyse it
                # this means you will add this file to a list of files: files_oi (files of interest)
                x = file
                if x in excel_filename_list:
                    files_oi.append(file)
    return files_oi

def get_cell_id_for_file (file, date, cell_dict):
    file_num = file.replace("MKcellchar_", "")
    file_num = int( file_num.replace(".cfs", "") )
    day = date.replace("_", "")
    cell_id = f"{day}{cell_dict[file_num]}"
    return cell_id, file_num


# %%
"""
Here you can add the cellsheet you want to analyse and the directory where your files are located
Then the main function executes the analysis for all your files in that directory.
"""


# define which excel sheet to use
# it should contain the numbers of cellchar recordings
# /mnt/live1/kmikulik/Analysis/Cellsheet_Patcherei.xlsx"
CELLSHEET = "/mnt/live1/kmikulik/Analysis/Cellsheet_Patcherei_ohne hash.xlsx"

# set the directory where your files for analysis are located
DIRECTORY = "/mnt/live1/kmikulik/recordings/Analysis/"


def main(CELLSHEET, DIRECTORY):
    """
    This funtion iterates through all files of interest and performs the analysis of these files.
    At the end it adds the output of each iteration to the corresponding row of the excel sheet we create at the beginning.
    """

    # create a dictionary with file as key and add calculated values as values inside a for loop later on
    value_dict = {}

    # first we create a list of files we want to analyse
    # the input of this function is an excel sheet
    excel_filename_list = files_for_analysis(CELLSHEET)
    
    #then we create a dictionary of cell and slice numbers for each file we want to analyse 
    #this funciton returns a dictionary with files as keys and slice (x) and cell(y) number as values 
    cell_dict = create_cell_id(CELLSHEET)
    
    # creaete a dictionary of all files in the specified directory with dates (folders) as keys
    dictionary = file_dictionary(DIRECTORY)
    for date, files in dictionary.items():
        files_oi = []
        # loop through all files of each date and filter for files with the correct names: MKcellchar_xxx.cfs
        for file in files: 
            if re.search("cellchar.*.cfs", file) != None:
                # for every file of this date create a variable x
                # then check if x is in the list you created from the excel sheet
                # if the file is on that list, you want to analyse it
                # this means you will add this file to a list of files: files_oi (files of interest)
                x = file
                if x in excel_filename_list:
                    files_oi.append(file)
        #loop through files of interest
        for i in range(len(files_oi) ) : 
            file = files_oi [i]
            file_name = f"{DIRECTORY}/{date}/{file}"
            
            #Now that we have only one single file specified by file_name, we can start the analysis of that file:
            array_3d = read_as_3Darray(file_name)
            timeline = create_timeline_array()
            # set_variables(array_3d)
            # the function analyse_cellchar returns a tuple with all calculated values
            value_tuple = analyse_cellchar(date, file, timeline, array_3d)
                
                #after analysing we can now add all values to a dictionary (filenames will be the keys of the dictionary)
                #make a list that contains all values of a certain file
            value_list = list(value_tuple)
                #create a unique cell_id by adding date and slice and cell number (from cell_list) together into one string
            cell_id, file_num = get_cell_id_for_file(file, date, cell_dict) 

            value_list.insert(0, cell_id)
            value_list.insert(1, file_num)           
                #value_list.insert(1, cell_list [i] )
                # for each file (key of dictionary) we add the tuple as values to a dictionary
            for value in value_tuple:
                value_dict[file] = value_list
                    # this is the order of the values in the tuple:
                    # (rmp, hold, input_res, testcur, testvol, deltavol, deltacur, minV, stabV, Ih_vol, rheo, burstratio,
                    # interval1, interval2, thr_vol, AP_distance, slope, intercept, APamp, APhw, APriseT, APfallT, APmaxrise, AP_list)
            # except:
            #     pass
    
    dataframe = pd.DataFrame.from_dict(data=value_dict, orient="index", columns=["cell_id", "file", "rmp", "hold", "input_res", "testcur", " testvol", "deltavol",
                                                                                     "deltacur", "minV", "stabV", "Ih_vol", "rheo", "burstratio",
                                                                                     "interval1", "interval2", "thr_vol", "AP_distance", "slope", "intercept",
                                                                                     "APamp", "APhw",
                                                                                     "APriseT", "APfallT", "APmaxrise", "AP_list"])
    #We do not want the filenames as indices:
    # dataframe.index.name = 'File'
    # dataframe.reset_index(inplace=True)
    # dataframe = dataframe.drop(["File"], axis = "columns")
        # (rmp, hold, input_res, testcur, testvol, deltavol, deltacur, minV, stabV, Ih_vol, rheo, burstratio,
        # interval1, interval2, thr_vol, AP_distance, slope, intercept, APamp, APhw, APriseT, APfallT, APmaxrise, AP_list)


    return dataframe


dataframe = main(CELLSHEET, DIRECTORY)

dataframe.index.name = 'File'
dataframe.reset_index(inplace=True)
dataframe = dataframe.drop(["File"], axis = "columns")
    
dataframe.to_pickle("/mnt/live1/kmikulik/Analysis/cellchar_dataframe.pkl")

dataframe = pd.read_pickle("/mnt/live1/kmikulik/Analysis/cellchar_dataframe.pkl")

#cell_id = dataframe [ ["cell_id", "file"] ]

    