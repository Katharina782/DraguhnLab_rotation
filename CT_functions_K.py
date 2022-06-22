#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 15:06:27 2021

@author: cthome
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#from datasheet_mod_v01 import datasheet_mod_v01
#import pingouin as pg
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
from scipy.interpolate import interp1d




#%% Martin Both code
def nyquist_upsampling(data):

    import pandas as pd

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

#%% action potential analysis

def AP_upsample(x_time, y_voltage):
    y_voltage_ny = nyquist_upsampling(y_voltage)
    x_time_ny  = np.linspace(x_time[0], x_time[-1], len(y_voltage_ny))

    y_voltage_ny = y_voltage_ny[99:-100:1]
    x_time_ny= x_time_ny[99:-100:1]
    
    return [x_time_ny,y_voltage_ny]

def determine_threshold (x_time, y_voltage):
    """ determine the threshold of the AP. """
    thr_section = np.nan
    x_time_ny, y_voltage_ny  = AP_upsample(x_time, y_voltage)
    dat_ds = np.diff(y_voltage_ny)/np.median( np.diff(x_time_ny) )
    
    ind_thr = np.nonzero( dat_ds > 20 )[0]
    if len(ind_thr) > 0:
        thr_section = y_voltage_ny[ind_thr[0]-1]

    return thr_section

def determine_rapidness(x_time, y_voltage):
    """ determine the onset rapidness of the AP. """
    slope = np.nan
    #x_time, y_voltage = AP_upsample(x_time, y_voltage)
    dat_ds = np.diff(y_voltage)/np.median( np.diff(x_time) )

    indrap = []
    for i in range(0,len(dat_ds)):
        if dat_ds[i] >= 15:
            indrap.append(i)
        if dat_ds[i] >= 50:
            indrap.append(i)
            break
    slope, intercept = np.polyfit(y_voltage[indrap],dat_ds[indrap],1)
    return [slope,intercept]

def determine_amplitude(x_time, y_voltage):
    """ determine the amplitude of the AP. """
    APamp = np.nan
    #x_time, y_voltage = AP_upsample(x_time, y_voltage)
    low = determine_threshold(x_time, y_voltage)
    peakind, peaks= find_peaks(y_voltage,height=0)
    up = peaks['peak_heights'][0]
    APamp = up-low
    return APamp

def determine_halfwidth(x_time, y_voltage):
    """ determine the halfwidth of the AP. """
    APhw = np.nan
    #x_time, y_voltage = AP_upsample(x_time, y_voltage)    
    APamp = determine_amplitude(x_time, y_voltage)
    APthes = determine_threshold(x_time, y_voltage)
    hAPamp = APthes + (APamp/2)
    firsthind = np.where(y_voltage >= hAPamp)[0][0]
    lasthind = np.where(y_voltage >= hAPamp)[0][-1]
    APhw = x_time[lasthind]-x_time[firsthind]
    return APhw
    
def determine_APriseT(x_time, y_voltage):
    """ determine the halfwidth of the AP. """
    APriseT = np.nan
    #x_time, y_voltage = AP_upsample(x_time, y_voltage)
    APamp = determine_amplitude(x_time, y_voltage)
    APthes = determine_threshold(x_time, y_voltage)
    AP10perc = APthes + (0.1*APamp)
    AP90perc = APthes + (0.9*APamp)
    firsthind = np.where(y_voltage >= AP10perc)[0][0]
    lasthind = np.where(y_voltage >= AP90perc)[0][0]
    APriseT = x_time[lasthind]-x_time[firsthind]  
    return APriseT

    
def determine_APfallT(x_time, y_voltage):
    """ determine the halfwidth of the AP. """
    APfallT = np.nan
    #x_time, y_voltage = AP_upsample(x_time, y_voltage)
    APamp = determine_amplitude(x_time, y_voltage)
    APthes = determine_threshold(x_time, y_voltage)
    AP10perc = APthes + (0.1*APamp)
    AP90perc = APthes + (0.9*APamp)
    firsthind = np.where(y_voltage >= AP90perc)[0][-1]
    lasthind = np.where(y_voltage >= AP10perc)[0][-1]
    APfallT = x_time[lasthind]-x_time[firsthind]  
    return APfallT

def determine_APmaxrise(x_time, y_voltage):
    """ determine the halfwidth of the AP. """
    APmaxrise = np.nan
    #x_time, y_voltage = AP_upsample(x_time, y_voltage)
    dat_ds = np.diff(y_voltage)/np.median( np.diff(x_time) )
    APmaxrise = dat_ds.max()
    return APmaxrise


#%% cellchar specific analysis

def cellchar_baseL(x_time, y_voltage, y_current):
    """ determine the halfwidth of the AP. """
    I_baseL = np.nan
    V_baseL = np.nan
    Rs_baseL = np.nan
    timeframeforbaseline = np.where(x_time <= 250)[0]
    I_baseL = np.mean(y_current[timeframeforbaseline])
    V_baseL = np.mean(y_voltage[timeframeforbaseline])
    
    timeframeforRs_baseL = np.where(  (x_time >= 350) & (x_time <= 395) )
    deltaV_Rs_baseL = np.mean(y_voltage[timeframeforRs_baseL]) - V_baseL
    deltaI_Rs_baseL = np.mean(y_current[timeframeforRs_baseL]) - I_baseL
    Rs_baseL = 1000 * (deltaV_Rs_baseL / deltaI_Rs_baseL)
    
    return [I_baseL,V_baseL,Rs_baseL]

def cellchar_Ih(x_time, y_voltage, y_current):
    """ determine the Ih parameters of stimulus protocol. """
    '''1. amplitude of current injected
       2. minimum of voltage deflection
       3. stable voltage deflection
       4. Ih quotient (peak/stable voltage)  '''
       
    deltaI_stim = np.nan
    timeframeforbaseline = np.where(x_time <= 250)[0]
    I_baseL = np.mean(y_current[timeframeforbaseline])
    V_baseL = np.mean(y_voltage[timeframeforbaseline])
    
    timeframeforstim = np.where(  (x_time > 800) & (x_time < 1300) )
    timeframeforstabV = np.where(  (x_time > 1100) & (x_time < 1300) )
    deltaI_stim = np.mean(y_current[timeframeforstim]) - I_baseL # strength of I injection
    deltaminV_stim = np.min(y_voltage[timeframeforstim]) - V_baseL # negative peak for Ih
    deltastabV_stim = np.mean(y_current[timeframeforstabV]) - V_baseL # stable potential for Ih
    Ih_peakQstab = deltaminV_stim / deltastabV_stim
    
    return [deltaI_stim, deltaminV_stim, deltastabV_stim ,Ih_peakQstab]

def cellchar_APpattern(x_time, y_voltage, y_current):
    """ determine the firing patter of the stimulation train. """
    ''' 1. amplitude of current injected, 
        2. number of AP
        3. indices of AP peaks
        4. time of AP peaks
        5. delay to first AP
        6. burstiness (bursty if > 1.6)  '''
        
    deltaI_stim = np.nan
    nAP = np.nan
    ind_peak = np.nan
    APtimes = np.nan
    firstAPtime = np.nan
    burstratio = np.nan
    
    timeframeforbaseline = np.where(x_time <= 250)[0]
    timeframeforstim = np.where(  (x_time > 800) & (x_time < 1300) )
    I_baseL = np.mean(y_current[timeframeforbaseline])
    deltaI_stim = np.mean(y_current[timeframeforstim]) - I_baseL # strength of I injection
    ind_peak, V_peaks = find_peaks(y_voltage,height=0)
    nAP = len(ind_peak)
    APtimes = x_time[ind_peak]
    firstAPtime = APtimes[0]-800
    if nAP > 4:
        ISIa = APtimes[3]-APtimes[2]
        ISIb = APtimes[1]-APtimes[0]
        burstratio = ISIa / ISIb
    
    return [deltaI_stim,nAP,ind_peak,APtimes,firstAPtime,burstratio]

#%% to evaluate the sinus protocol



def resample(array, npts):
    from scipy.interpolate import interp1d
    interpolated = interp1d(np.arange(len(array)), array, axis = 0, kind='cubic', fill_value = 'extrapolate')
    downsampled = interpolated(np.linspace(0, len(array), npts))
    return downsampled

def function1d(x_array, y_array):
    from scipy.interpolate import interp1d
    interpolated = interp1d(x_array, y_array, fill_value = 'extrapolate')
    return interpolated


def determine_smpfac(x_time):
    smpfac = np.nan
    smpfac = x_time[2] - x_time[1]
    smpfac = round(smpfac,5)
    return smpfac

def normalizeSIN(y_current):
    y_currentN1 = (y_current - np.mean(y_current)) / ((np.max(y_current)-np.min(y_current))/2)
    return y_currentN1


def determine_Ifreq(x_time, y_current):
    Ifreq = np.nan
    smpfac = determine_smpfac(x_time)
    y_currentN1 = normalizeSIN(y_current)
    #Savitzky-Golay filter = digital filter to smooth data wihtout distorting the signal tendency
    #y_currentNT1 is the data to be filtered
    y_currentN1 = savgol_filter(y_currentN1,51,3)
    #take all normalized current values of 500ms to 1000ms
    y_currentN1 = y_currentN1[round(500/smpfac):round(1000/smpfac)]
    zcrosses = np.where(np.diff(np.sign(y_currentN1)))[0]
    Ifreq = len(zcrosses)
    return Ifreq

def phasesize(x_time,y_current):
    phasesizeind = np.nan
    phasesizeind = (1000/ determine_Ifreq(x_time, y_current)) / (determine_smpfac(x_time))
    return phasesizeind

def phasestarts(y_current):
    phasestartsind = np.nan
    y_currentN1 = normalizeSIN(y_current)
    y_currentN1 = savgol_filter(y_currentN1,51,3)    
    phasestartsind  = np.where(np.diff(np.sign(y_currentN1))  > 1)[0]
    phasestartsind = phasestartsind[np.where(y_currentN1[phasestartsind+100] > 0 )]
    return phasestartsind 

def total_number_cycles(x_time, y_current):
    Ifreq = np.nan
    smpfac = determine_smpfac(x_time)
    y_currentN1 = normalizeSIN(y_current)
    #Savitzky-Golay filter = digital filter to smooth data wihtout distorting the signal tendency
    #y_currentNT1 is the data to be filtered
    y_currentN1 = savgol_filter(y_currentN1,51,3)
    #take all normalized current values of 500ms to 1000ms
    y_currentN1 = y_currentN1[round(0/smpfac):round(1000/smpfac)]
    zcrosses = np.where(np.diff(np.sign(y_currentN1)))[0]
    cyclenum = (len(zcrosses)) -1
    print(cyclenum)
    return cyclenum

def determine_freq (x_time, y_current):
    current_1sec = y_current[0:5000]
    wavepeaks = np.where(np.diff(current_1sec) == 0)[0]
    wavepeaks = np.where(np.max(np.diff))[0]
    print(wavepeaks)
    len(wavepeaks)

for i in range(1,cyclenum +1):
    cyclecount.append(i)
print(cyclecount)
np.repeat()
cycletime = 1/Ifreq
degree = np.arange(0,361,1)
phase = list()
x = 180
for i in range(0,cyclenum:
    if x =< 360:           
        x = 180 + i
    else:
        x = 0

x= np.repeat(degree, cyclenum)
x_time = np.arange(180,360,cycletime)

#phases for the first cycle
for i in range (y_current8):
    
    
    
phaselist = list()    
x = 180    
count = 0
for i in range(0, 100000):
    if 0 =< x =< 360:
        count = count + 1
        x = 180 + count
        phaselist.append(x)
    else: 
        x = 0
        
def determine_frequency 
    Ifreq = np.nan
    y_currentN1 = normalizeSIN(y_current)
    #Savitzky-Golay filter = digital filter to smooth data wihtout distorting the signal tendency
    #y_currentNT1 is the data to be filtered
    y_currentN1 = savgol_filter(y_currentN1,51,3)
    #take all normalized and filtered current values of 500ms to 1000ms
    y_currentN1 = y_currentN1[round(500/smpfac):round(1000/smpfac)]
    #the function np.sign gives -1 if x < 0 and +1 if x > 0
    #since np.diff calculates the difference between two adjacent values this will be 0 if 
    # numbers are only -1 or 1.
    #However at the position where the y axis is crossed you will have +1 - (-1)
    #therefore, at the points where the y axis is crossed you will get a value different from 0
    zcrosses = np.where(np.diff(np.sign(y_currentN1)))[0]
    Ifreq = len(zcrosses)
    return Ifreq

    
    