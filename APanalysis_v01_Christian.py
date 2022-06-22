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

#%% martin both code
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
    global slope, intercept
    """ determine the onset rapidness of the AP. """
    slope = np.nan
    #x_time, y_voltage = AP_upsample(x_time, y_voltage)
    #dat_ds = slope (mv/mS)
    dat_ds = np.diff(y_voltage)/np.median( np.diff(x_time) )

    indrap = []
    for i in range(0,len(dat_ds)):
        if dat_ds[i] >= 15:
            indrap.append(i)
        if dat_ds[i] >= 50:
            break
    #indrap is a list of indices of those slope values of the action potential curve that are between 15 and 50
    #polyfit fits a polynomial to points x (y_voltage), y (dat_ds)
    #last number = degree 1 -> linear polynomial
    #y_voltage[indrap] creates a list of the voltage values where slope is between 15 and 50
    #dat_ds[indrap] creates a list of the slope values at the before determined indices indrap
    #the list of values for voltage and slope are fitted to a linear polynomial 
    #polyfit returns the coefficient(slope of the linear function) and the y-intercept
    slope, intercept = np.polyfit(y_voltage[indrap],dat_ds[indrap],1)
    return [slope,intercept]

def determine_amplitude(x_time, y_voltage):
    """ determine the amplitude of the AP. """
    APamp = np.nan
    #x_time, y_voltage = AP_upsample(x_time, y_voltage)
    low = determine_threshold(x_time, y_voltage)
    peakind, peaks= find_peaks(y_voltage,height=0)
    #das zieht nur den Wert des peaks aus der Liste, weil die Liste nur einen value hat -> ersten index [0] 
    up = peaks['peak_heights'][0]
    APamp = up-low
    return APamp

def determine_halfwidth(x_time, y_voltage):
    """ determine the halfwidth of the AP. """
    APhw = np.nan
    #x_time, y_voltage = AP_upsample(x_time, y_voltage)    
    APamp = determine_amplitude(x_time, y_voltage)
    APthes = determine_threshold(x_time, y_voltage)
    #if we divide the amplitude by 2, we have half the distance between threshold and peak.
    #if we add this distance to the threshold, we get the voltage value at this half height
    hAPamp = APthes + (APamp/2)
    #np.where will print an array containing a list of all indices at which y-voltage satisfies the condition
    #with the first [0] we get the list at the first index of the array 
    #with the second [0] we get the first index value of the list -> this is the point on the left flank of the cure
    firsthind = np.where(y_voltage >= hAPamp)[0][0]
     #by taking the last value of the list, we get the point on the right flank of the curve
    lasthind = np.where(y_voltage >= hAPamp)[0][-1]
    #now we subtract the time values at the left flank point and the right flank point in order to get the halfwidth
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

    #%% import stuff
# efile1 = '/mnt/archive1/cthome/Backup_ProjectArchive/Y_AISdiversity_PYR/Kathi/' +\
#     '23_02_2021/MKcellchar_021.cfs'
# efile2 = '/mnt/archive1/cthome/Backup_ProjectArchive/Y_AISdiversity_PYR/Kathi/' +\
#     '23_02_2021/MKsinus_013.cfs'
#efile1 = '/mnt/live1/cthome/signal/MKcellchar_021.cfs'
efile1 = "/mnt/live1/kmikulik/recordings/23_02_2021/MKcellchar_021.cfs" #
#efile1 = "/mnt/live1/kmikulik/recordings/23_02_2021/MKcellchar021_cfs"

   #%%define stuff 

rec1 = stfio.read(efile1,"tcfs")
rec2=np.array(rec1)
smpfac = 0.02
timeline = np.arange(0,2000,smpfac).tolist()

#%% figure for illustration of problem
#We want the phase in which every individual AP fires.
#for now peak is enough, but later threshold.

x= rec2[1][22]#[]
x2 = rec2[0][22]#[]
peaks, peakind = find_peaks(x,height=0)
fig01, axs = plt.subplots(2) # sharey = sharing of y axis beschriftung
fig01.suptitle('sinus protocol')
axs[0].plot(timeline,x)
axs[0].plot(peaks*smpfac, x[peaks],'x')
axs[0].plot(timeline, np.zeros_like(x),'--',color='gray')
axs[0].set_xlabel('time (ms)')
axs[0].set_ylabel('voltage (mV)')
axs[1].plot(timeline, x2)


#%% get threshold and onset rapidness of first AP in trace 22
#define the area around the peak that we analyse:
    
y_voltage = rec2[1][22][(peaks[0]-100):peaks[0]+150]
x_time = timeline[(peaks[0]-100):peaks[0]+150]

determine_rapidness(x_time,y_voltage)

#x_time, y_voltage = AP_upsample(x_time, y_voltage)
dat_ds = np.diff(y_voltage)/np.median( np.diff(x_time) )

# determines voltage range for plotting rapidness
thresllist = [determine_threshold(x_time,y_voltage)-2, 
              determine_threshold(x_time,y_voltage),
              determine_threshold(x_time,y_voltage)+5]
#list of 3 voltage values
APrapidness = determine_rapidness(x_time, y_voltage)
abline_values = [APrapidness[0] * i + APrapidness[1] for i in thresllist]


fig02, axs = plt.subplots(2,1, figsize=(6,6), sharex=False) # sharey = sharing of y axis beschriftung
fig02.suptitle('sinus protocol')

axs[0].plot(x_time, y_voltage)
axs[0].axhline(y=determine_threshold(x_time,y_voltage), color='r', linestyle='-')
axs[0].set_xlabel('time (ms)')
axs[0].set_ylabel('voltage (mV)')

#[0:-1], geht bis zum vorletzten Wert
axs[1].plot(y_voltage[0:-1], dat_ds)
axs[1].axhline(y=20, color='r', linestyle='-')
axs[1].plot(thresllist, abline_values, 'b')
axs[1].set_xlabel('voltage (mV)')
axs[1].set_ylabel('slope (mV/ms)')

fig03, axs = plt.subplots(1,1, figsize=(6,6), sharex=False) # sharey = sharing of y axis beschriftung
fig03.suptitle('results')
plt.text(0.1,0.9,'AP v-thrshld: '+"{:.2f}".format(determine_threshold(x_time,y_voltage)))
plt.text(0.1,0.8,'AP ons rapid: '+"{:.2f}".format(determine_rapidness(x_time,y_voltage)[0]))
plt.text(0.1,0.7,'AP amplitude: '+"{:.2f}".format(determine_amplitude(x_time,y_voltage)))
plt.text(0.1,0.6,'AP halfwidth: '+"{:.2f}".format(determine_halfwidth(x_time,y_voltage)))
plt.text(0.1,0.5,'AP max slope: '+"{:.2f}".format(determine_APmaxrise(x_time,y_voltage)))
plt.text(0.1,0.4,'AP rise time: '+"{:.2f}".format(determine_APriseT(x_time,y_voltage)))
plt.text(0.1,0.3,'AP fall time: '+"{:.2f}".format(determine_APfallT(x_time,y_voltage)))
#%%plt.text(0.1,0.1,'v threshold: '+str(determine_threshold(x_time,y_voltage)))
