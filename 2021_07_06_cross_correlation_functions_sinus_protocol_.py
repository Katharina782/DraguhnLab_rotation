#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 08:21:08 2021

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







#%% Martin Both code
def nyquist_upsampling(data):


    dummy  = pd.read_csv('/mnt/live1/mboth/PythonWS/Interpolation_kernel.csv')
    kernel = dummy.to_numpy()
#    d_u = 0

    up_fac = 10;
#    dum = np.sin( np.arange(1,100,1) ) # the originatl data
    dum = data

    num_zeros = up_fac-1 # upFac-1
    dum_u = np.reshape(  np.concatenate( ( np.array([dum]), np.zeros([num_zeros, len(dum)]) ) ).T, ( 1, (num_zeros+1)*len(dum) )  )[0] #upsample in matlab

    nm = len(kernel)
    nx = len(dum_u)
    if nm % 2 != 1:    # n even
        m = nm/2
    else:
        m = (nm-1)/2

    X = np.concatenate( (np.ones(int(m))*dum_u[0], dum_u, np.ones(int(m))*dum_u[-1]) )
    y = np.zeros(nx)

    indc = np.repeat( np.expand_dims(  np.arange(0,nx,1), axis=1  ), nm, axis=1 ).T
    indr = np.repeat( np.expand_dims(  np.arange(0,nm,1), axis=1  ), nx, axis=1 )
    ind = indc + indr

    # xx = X(ind) this is what you do in matlab: use the array of indices to construct a new array with the values at the indices
    xx = np.zeros(ind.shape)
    for cX in range(0, nm):
        for cY in range(0, nx):
            xx[cX,cY] = X[ind[cX][cY]]

    y = np.inner(kernel.T, xx.T)[0].T

    return y


"""
If the current injection changes during the frame we want to declare this frame as invalid.
We check at 3 different intervals in the frame. If the maximum current value differs more than 10 pA
between these intervals the frame is invalid.
"""
def invalid_frame(rec, i):
    cur = rec[0]
    curi = cur[i]
    max1 = np.max( curi[0:10000] )
    max2 = np.max( curi[50000:60000] )
    max3 = np.max( curi[90000:100000] )
    if abs(max1 - max2) > 10:
        return False
    if abs(max2 -max3) > 10:
        return False
    if abs(max1 - max3) > 10:
        return False
    else: 
        return True
    
"""
write a function that checks if the current injection in a frame is constant or not.
If the current is not constant over a frame the invalid_frames function will tell us. 
If the fram is valid we add it to our list of frames for analysis
"""
def discard_frames(rec):
    frames = len(rec[0])
    frames_for_analysis = list()
    frames_not_for_analysis = list()
    for i in range (0, frames):
        valid_frame = invalid_frame(rec, i) 
        if valid_frame == False:
            frames_not_for_analysis.append(i)
        else:
            frames_for_analysis.append(i)
    # print(frames_not_for_analysis)
    # print(frames_for_analysis)
    return frames_for_analysis
        
#%%     

def normalizeSIN(y_current):
    y_currentN1 = (y_current - np.mean(y_current)) / ((np.max(y_current)-np.min(y_current))/2)
    return y_currentN1


def determine_frequency (y_current, x_time):
    """
    this function will determine the frequency of your current injection
    No upsampling performed yet.
    """
    Ifreq = np.nan
    y_currentN1 = normalizeSIN(y_current)
    #Savitzky-Golay filter = digital filter to smooth data wihtout distorting the signal tendency
    #y_currentNT1 is the data to be filtered
    y_currentN1 = savgol_filter(y_currentN1,51,3)
    #take all normalized and filtered current values of 500ms to 1000ms
    y_currentN1 = y_currentN1[round(500/0.02):round(1000/0.02)]
    #y_currentN1 = resample(y_currentN1, 2000)
    #the function np.sign gives -1 if x < 0 and +1 if x > 0
    #since np.diff calculates the difference between two adjacent values this will be 0 if 
    # numbers are only -1 or 1.
    #However at the position where the y axis is crossed you will have +1 - (-1)
    #therefore, at the points where the y axis is crossed you will get a value different from 0
    zcrosses = np.where(np.diff(np.sign(y_currentN1)))[0]
    #zcrosses = (np.diff(np.sign(y_currentN1))).nonzero()[0]
    Ifreq = len(zcrosses)
    if Ifreq < 4:
        Ifreq = 0
    if (Ifreq >= 4) & (Ifreq < 10):
        Ifreq = 5
    # if Ifreq != 5:
    #     pass
    # elif Ifreq != 20:
    #     pass
    # elif Ifrq != 50:
    #     pass
    # elif Ifreq != 200:
    #     pass
    # elif Ifreq != 500:
    #     pass
    # else:
    print(Ifreq)
        
    return Ifreq


        

                              

"""
For every valid frame determine the frequency of the current injection.
For this we iterate through the list with valid frames and use the function determine_frequency()
"""
def frequency_per_frame(rec, frames_for_analysis, x_time):
    freq_frame = {}
    cur = rec[0]
    for frame in frames_for_analysis:
        cur_frame = cur[frame]
        freq_frame[frame] = determine_frequency(cur_frame, x_time)
    return freq_frame





""" 
We want to know the index of each peak in each frame.
For this we create a dictionary where all APs (values) are assigned to one frame (key).
"""
def APs_per_frame(rec, frames_for_analysis):
    vol = rec[1]
    peak_frame = {}
    for frame in frames_for_analysis:
        voli = vol[frame]
        peak_list = []
        peaks, _ = find_peaks(voli, height = 0, prominence = 1) 
        for peak in peaks:
            peak_list.append(peak)
            peak_frame[frame] = peak_list
    return peak_frame

"""
Now we need the voltage intervals around all peaks.
We use the peak_frame dictionary to get each peak per frame and define the voltage interval around it.
We want to get back a dictionary that assigns each voltage interval around a peak to a list 
# and then assigns the list to the corresponding frame

"""
def find_AP_intervals(rec, peak_frame):
    vol = rec[1]
    ap_vol = {}
    ap_time = {}
    for frame, peaks in peak_frame.items():
        vol_int_frame = []
        time_int_frame = []
        for peak in peaks:
            vol_per_peak = vol[frame][(peak - 100) : (peak + 150) ]
            time_per_peak = timeline[ (peak - 100) : (peak + 150) ]
            #add values to a dictionary
            vol_int_frame.append(vol_per_peak)
            time_int_frame.append(time_per_peak)
        ap_vol[frame]= vol_int_frame
        ap_time[frame] = time_int_frame
    return ap_vol, ap_time


"""
we create a dictionary with a list of timepoints of threshold of APs for each frame. 
We then want to know the phase in which that threshold is fired
"""
def determine_APthrestime(x_timeAP,y_voltageAP):
    APthrestime = np.nan
    y_voltageN1 = savgol_filter(y_voltageAP,11,3)  
    #we get a voltage value for the threshold from the function determine_threshold
    APthres = determine_threshold(x_timeAP, y_voltageN1)
    if APthres == "undefined":
        APthrestime = "undefined"
    else:
        APthresind  = np.where(y_voltageN1 > APthres)[0][0]
        APthrestime = x_timeAP[APthresind]
    return APthrestime


    # APthrestime = np.nan
    # y_voltageN1 = savgol_filter(vol_int,11,3)  
    # #we get a voltage value for the threshold from the function determine_threshold
    # APthres = determine_threshold(time_int, y_voltageN1)
    # APthresind  = np.where(y_voltageN1 > APthres)[0][0]
    # APthrestime = time_int[APthresind]
# #test
# thr_list = []
# for frame, vol in ap_vol.items():
#     for i in range(0, len(vol)):
#         x_timeAP = ap_time[frame][i]
#         y_voltageAP = ap_vol[frame][i]
#         thr = determine_APthrestime(x_timeAP, y_voltageAP)
#         thr_list.append(thr)


"""
since we determine the thrshold of the action potential with this function we need to be careful in case there is no value for dat_ds > 20
if this is the case we will get an error message, since ind_thr[0] will be out of bound if ind_thr is an empty array

"""
def determine_threshold (x_time, y_voltage):
    """ determine the threshold of the AP. """
    thr_section = np.nan
    x_time_ny, y_voltage_ny  = AP_upsample(x_time, y_voltage)
    dat_ds = np.diff(y_voltage_ny)/np.median( np.diff(x_time_ny) )
    #only if dat_ds is bigger than 20 the array will contain True values 
    #nonzero will only take those values where dat_ds is True and put them into an array
    #the threshold index will be the first value where dat_ds > 20
    ind_thr = np.nonzero( dat_ds > 20 )[0]
    if len(ind_thr) > 0:
        thr_section = y_voltage_ny[ind_thr[0]-1]
    else: 
        thr_section = "undefined"

    return thr_section


    # thr_section = np.nan
    # x_time_ny, y_voltage_ny  = AP_upsample(time_int, vol_int)
    # dat_ds = np.diff(y_voltage_ny)/np.median( np.diff(x_time_ny) )
    
    # ind_thr = np.nonzero( dat_ds > 20 )[0]
    # if len(ind_thr) > 0:
    #     thr_section = y_voltage_ny[ind_thr[0]-1]
    # else: 
    #     thr_section = "undefined"



def AP_upsample(x_time, y_voltage):
    y_voltage_ny = nyquist_upsampling(y_voltage)
    x_time_ny  = np.linspace(x_time[0], x_time[-1], len(y_voltage_ny))

    y_voltage_ny = y_voltage_ny[99:-100:1]
    x_time_ny= x_time_ny[99:-100:1]
    
    return [x_time_ny,y_voltage_ny]


def find_thresholds_per_frame(ap_vol, ap_time):
    frame_thr = {}
    #peak_thr_distance = {}
    for frame, peaks in ap_vol.items():
        thr_list = []
        for peak in range(0, len(peaks)):
            # print(peak)
            # print(frame)
            if len ( ap_vol [frame] [peak ] ) > 0 : 
                vol_int = ap_vol[frame][peak]
                time_int = ap_time[frame][peak]
                thr_time = determine_APthrestime(time_int, vol_int)
            else: 
                thr_time = "undefined"
            # vol_int_up = AP_upsample_vol(vol_int)
            # dat_ds = np.diff (vol_int_up) / 0.002
            # ind_thr = np.nonzero(dat_ds > 20)[1]
            # if len(ind_thr) > 0:
            #     thr_section = vol_int_up[0][ind_thr[0] - 1]
            #     thr = ind_thr[0] -1
            thr_list.append(thr_time)
            # peaks, _ = find_peaks(vol_int_up[0], height = 0, prominence = 1)
            # distance = peaks[0] - thr
            # dist_list.append(distance)
        frame_thr[frame] = thr_list
        #peak_thr_distance[frame] = dist_list
    return frame_thr





rec = read_file(file)
timeline = create_timeline_array()


frames_for_analysis = discard_frames(rec)
freq_frame = frequency_per_frame(rec, frames_for_analysis, timeline)
#period = period(freq_frame)
peak_frame = APs_per_frame (rec, frames_for_analysis)
ap_vol, ap_time = find_AP_intervals(rec, peak_frame)
frame_thr = find_thresholds_per_frame(ap_vol, ap_time)




"""
plot the recordings
"""
vol = rec[1]
cur = rec[0]
frame_number = len(vol)

for i in range(0, frame_number):
    fig01, ax1 = plt.subplots(2)
    fig01.suptitle(i)
    ax1[0].plot(timeline, vol[i])
    ax1[0].set_ylim([-120,50])
    ax1[0].set_xlabel("time [ms]")
    ax1[0].set_ylabel("voltage [mV]")
    ax1[1].plot(timeline, cur[i])
    ax1[1].set_ylim([-200,500])
    ax1[1].set_ylabel("current [pA]")
    #find peaks returns ndarray -> indices of peaks that satisfy all given conditions
    peaks, _ = find_peaks(vol[i], height = 0)  
    for peak in peaks:
        ax1[0].axvline(x = timeline[peak])


plt.title(i)
plt.axis([0, 2000, -120, 50])
plt.xlabel("time [ms]")
plt.ylabel("potential [mV]")
plt.show()
    
    
    
    
    
    
    
    
    
"""
this function takes cur & vol trace of one frame as an input  
it returns a dictioinary with frame as key and the delay in  ms as values
"""
def cross_correlation(rec, frames_for_analysis):
    membrane_delay = {}
    cur = rec[0]
    vol = rec[1]
    for frame in frames_for_analysis:
        print(frame)
        voli = vol[frame]
        curi = cur[frame]
        #mode = "full" -> cross correlation at each point
        #The best fit, is where correlation factor is the highest
        #returns result for every time point where a and v have some overlap 
        #output is 199 999, because the result of a cross correlation of data A and B is a vector with lenght(A) + length (B) - 1
        crosscorr = np.correlate(curi, voli, "full")
        #the function returns an array of correlations
        #get the maximum value OR minimum value when values of crosscorrelation are negative
        #this is the point in time where the signals are best aligned:
        if crosscorr[100000] > 0:
            max_ind = np.argmax(crosscorr)
        else: 
            max_ind = np.argmin(crosscorr)
            
        print(max_ind)
        #max_val = crosscorr[max_ind]
    
        #by calculating the positive or negative difference between a perfect alignment of the curve and alignment of all data points
        # for alignement of all data points you need to 100 000
        #you get the number of datapoints by which the voltage signal lags behind or leads
        #if the vol trace is leading, then the delay value will be negative
        #if the vol trace is lagging, then the delay value will be positive
        #in this case since the lag is negative the vol trace is leading and the current trace is lagging
        ind_delay = max_ind - 100000
        time_delay = ind_delay*0.02
        membrane_delay [frame] = time_delay
        print(time_delay)
    return membrane_delay

# crosscorr = np.correlate(cur2, vol2, "full")
# min_ind = np.argmin(crosscorr)
# ind_delay = min_ind - 100000
# time_delay = ind_delay * 0.02
# plt.plot(crosscorr)


def phase_delay (membrane_delay, frame, degree, period):
    #get time_delay from cross correlation function:
    time_delay = membrane_delay[frame]
    #how long does one degree last in ms?
    time_1degree = round( (period/360), 4)
    #phase delay in degrees:
    delay = time_delay/time_1degree
    degree_delayed = degree + delay
    if degree_delayed > 360:
        degree_delayed = degree_delayed -360
    if degree_delayed < 0:
        degree_delayed = 360 - abs(degree_delayed)
    else:
        degree_delayed = degree_delayed
    return degree_delayed




"""
Since the current sinus wave does not always start the same (not a sin wave at the beginning and not at 0 or 180°) 
we have to define the start of the first period. 
We define the start of the first period as the first time the curve crosses the x-axis from top to bottom.
Since it is a sinus wave the phase at this point of the period is 180
"""
def start_period (rec, frame):
    cur = rec[0]
    cur = cur[frame]
    curN1 = normalizeSIN(cur)
    curN1 = savgol_filter(curN1, 51, 3)
    #zcrosses = np.where ( np.diff ( np.sign( curN1[0:100] ))) [0]
    #np.sign returns an array with +1 in case the value is postive or -1 in case the value is negative
    #when the curve crosses the x-axis the value for np.diff (np-sign()) will be -2 or +2, else it will always be zero
    #when we cross from +1 to -1 np.diff will be -2 and the phase at this position will be 180°
    try:
        ind_180 = np.where (np.diff ( np.sign( curN1[0:100000]) )  == -2  )[0][0]
        time_180 = ind_180 * 0.02
    except:
        time_180 = "undefined"
    return time_180
    
#testdistance = 1143
# for 
# period = 200
# rest = distance % period
# percentage = ( rest/period )
# #add 0.5 to start at 0 degree and not 180 degree
# percentage = percentage + 0.5
# if percentage >= 1:
#     percentage = percentage - 1
# degree = percentage * 360

"""
Find indices where period starts for each frequency

"""

def find_period(freq_frame):
    """
    We want to know how long one phase lasts.
    To know how many ms one cycle lasts we divide 1000ms/frequency (eg.: 50)
    The result are x ms/cycle.
    To get the indices we divide the time in ms by 0.02.
    """
    periods = {}
    for frame, Ifreq in freq_frame.items():
        #for freq in Ifreq:
        if Ifreq == 0:
            periods [frame] = "undefined"
        else: 
            periods[frame] =  ((1000 / Ifreq))
            #period_time = period_time / (0.02)
            #period_list.append(period_time)
    return periods





"""
now we want to find the phase for each AP threshold
In case the threshold is undefined we also set the phase to undefined
"""
def find_phase (rec, frame_thr, freq_frame, frames_for_analysis):
    #time_180 = start_period(rec, frame)
    phase_test = {}
    membrane_delay = cross_correlation(rec, frames_for_analysis)
    for frame, thrs in frame_thr.items():
        #with this function we get the time point where the first period start with phase 180
        time_180 = start_period(rec, frame)
        degree_list = []

        for thr in thrs:
            if thr == "undefined":
                degree = "undefined"
                
            else: 
                APthr = thr
                #we calculate the distance in time between the threshold and the first period start (phase 180°)
                if time_180 == "undefined":
                    degree == "undefined"
                else: 
                    distance = APthr - time_180
                    #the find_period function returns an array with the period of each frame
                    #We take only the period of the frame we are currently accessing with the for loop
                    periods = find_period(freq_frame)
                    period = periods[frame]
                    if period == "undefined":
                        degree = "undefined"
                        degree_delayed = "undefined"
                    else:
                        # dividing distance by period we get the rest of the division.
                        # the rest represents the not yet finished period in which our threshold is located
                        rest = distance % period
                        #we calculate the percentage of the whole period 
                        percentage = ( rest/period )
                        # now we add 0.5 to start at 0 degree and not 180 degree
                        percentage = percentage + 0.5
                        if percentage >= 1:
                            percentage = percentage - 1
                        degree = round (percentage * 360)
                        degree_delayed = phase_delay(membrane_delay, frame, degree, period)
            degree_list.append([degree, degree_delayed])
        phase_test[frame] = degree_list
    return phase_test



phase_dict = find_phase(rec, frame_thr, freq_frame, frames_for_analysis)



"""
test why in the second file the crosscorr values are negative
"""
cur2 = rec[0][13]
vol2 = rec[1][13]
crosscorr = np.correlate(cur2, vol2, "full")
min_ind = np.argmin(crosscorr)
ind_delay = min_ind - 100000
time_delay = ind_delay * 0.02
plt.plot(crosscorr)


plt.plot(cur2[0 : 10000])
plt.plot(vol2[786:10000])


vol_int_cor = vol2[333: 1000]
min_vol_cor = min(vol2[333:1000])
min_ind_cor = np.where(vol_int_cor == min_vol_cor)[0][0]

cur_int_cor = cur2[0: 1000 - 333]
min_cur_cor = min(cur2[0:1000-333])
min_ind1 = np.where(cur_int_cor == min_cur_cor)[0][0]







cur = rec[0]
vol = rec[1]


voli = vol[24]
curi = cur[24]
#mode = "full" -> cross correlation at each point
#The best fit, is where correlation factor is the highest
#returns result for every time point where a and v have some overlap 
#output is 199 999, because the result of a cross correlation of data A and B is a vector with lenght(A) + length (B) - 1
crosscorr = np.correlate(curi, voli, "full")
#the function returns an array of correlations
#get the maximum value OR minimum value when values of crosscorrelation are negative
#this is the point in time where the signals are best aligned:
if crosscorr[500] > 0:
    max_ind = np.argmax(crosscorr)
else: 
    max_ind = np.argmin(crosscorr)
    
print(max_ind)
#max_val = crosscorr[max_ind]

#by calculating the positive or negative difference between a perfect alignment of the curve and alignment of all data points
# for alignement of all data points you need to 100 000
#you get the number of datapoints by which the voltage signal lags behind or leads
#if the vol trace is leading, then the delay value will be negative
#if the vol trace is lagging, then the delay value will be positive
#in this case since the lag is negative the vol trace is leading and the current trace is lagging
ind_delay = max_ind - 100000
time_delay = ind_delay*0.02
membrane_delay [frame] = time_delay
print(time_delay)