                                    #   %reset -f
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


# def cellchar_I_baseL(x_time, y_current):
#     """ determine the halfwidth of the AP. """
#     I_baseL = np.nan
#     timeframeforbaseline = np.where(x_time <= 250)[0]
#     I_baseL = np.mean(y_current[timeframeforbaseline])
#     return I_baseL

# def cellchar_V_baseL(x_time, y_voltage):
#     """ determine the halfwidth of the AP. """
#     V_baseL = np.nan
#     timeframeforbaseline = np.where(x_time <= 250)[0]
#     V_baseL = np.mean(y_voltage[timeframeforbaseline])
#     return V_baseL

# def cellchar_Rscntr(x_time, y_voltage, y_current):
#     """ determine the halfwidth of the AP. """
#     Rscntr = np.nan
#     timeframeforbaseline = np.where(x_time <= 250)[0]
#     I_baseL = np.mean(y_current[timeframeforbaseline])
#     V_baseL = np.mean(y_voltage[timeframeforbaseline])
    
#     timeframeforRscntr = np.where(  (x_time >= 350) & (x_time <= 395) )
#     deltaV_Rscntr = np.mean(y_voltage[timeframeforRscntr]) - V_baseL
#     deltaI_Rscntr = np.mean(y_current[timeframeforRscntr]) - I_baseL
#     Rscntr = 1000 * (deltaV_Rscntr / deltaI_Rscntr)
#     return Rscntr

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

    #%% import stuff

efile1 = '/mnt/live1/cthome/signal/MKcellchar_021.cfs'
 

   #%%define stuff 

rec1 = stfio.read(efile1,"tcfs")
rec2=np.array(rec1)
smpfac = 0.02
x_time = np.arange(0,2000,smpfac).tolist()

#Here we define the timeline as an array, so we can access it with the functions defined where the input will be x_time
x_time = np.asarray(x_time)

#%% figure for illustration of problem
#We want the phase in which every individual AP fires.
#for now peak is enough, but later threshold.

y_voltage = rec2[1][22]#[]
y_current = rec2[0][22]#[]
#this only looks at the peaks of frame 22
peaks, peakind = find_peaks(y_voltage,height=0)
fig01, axs = plt.subplots(2) # sharey = sharing of y axis beschriftung
fig01.suptitle('sinus protocol')
axs[0].plot(x_time,y_voltage)
axs[0].plot(peaks*smpfac, y_voltage[peaks],'x')
axs[0].plot(x_time, np.zeros_like(y_voltage),'--',color='gray')
axs[1].plot(x_time, y_current)


cellchar_I_baseL(x_time, y_current)
cellchar_V_baseL(x_time, y_voltage)
cellchar_Rscntr(x_time, y_voltage, y_current)

#%% get threshold and onset rapidness of first AP in trace 22
