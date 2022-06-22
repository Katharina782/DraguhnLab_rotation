#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 12:12:15 2021

@author: kmikulik
"""

#%%import libraries
# Import libraries.
import numpy as np
import pandas as pd
import seaborn as sns
sns.set(style="ticks", palette='rocket')

import matplotlib.pyplot as plt
from matplotlib import rc
rc("pdf", fonttype=42)

#from sinaplot import  sinaplot
def cm2inch(value):
    return value/2.54
    
import stfio
import re

from scipy.signal import find_peaks
import math

#from scipy.optimize import curve_fit
import os
import errno

from os import listdir

#writing to an excel sheet 
import xlwt

#%%
        
def read_as_3Darray(file_name):
    global array_3d
    rec = stfio.read(file_name,"tcfs")
    #convert data into an array
    array_3d = np.array(rec)
    #np.arrange creates arrays with regularly incrementing values
    #first value is start of array, second is end of array
    #in this case 0.02 will be the increment
    

def create_timeline_array():
    global timeline        
    #create timeline as an x-axis
    #np.arrange creates arrays with regularly incrementing values
    #first value is start of array, second is end of array
    #in this case 0.05 will be the increment -> this is to be able to use whole numbers for calculations
    #this timeline cannot be used for plotting voltage or current, because it only has 60000 values instead of 60606
    timeline = np.arange(0,3000,0.05).tolist()
    timeline = np.asarray(timeline)
    
def create_timeline_plots():
    global xtime
    #this timeline can be used for creating plots, since it has 60 606 values, like the vol trace
    xtime = np.arange(0, 3000, (3000/60606))
    xtime = np.asarray(xtime)
    

#%% set_parameters

def voltage_channel(array_3d):
    global vol
    #you can access the first dimension of the 3d array
    #the votlage channel is the 3rd in the recording -> 2nd in python
    #vol will then be a 2d array with one dimension conatianing the frames and the second containing datapoints
    vol = array_3d[2] 
    


def current_channel(array_3d):
    global cur
    #the current channel is the 1st in the recording -> zero in python
    cur = array_3d[0]
 


def first_voltage_frame(array_3d):
    global vol0
    #by acocessing the 2d array vol -> you can get the voltage values of the first frame of the file
    vol = array_3d[2]
    vol0 = vol[0]



def first_current_frame(array_3d):
    global cur0
    cur = array_3d[0]
    cur0 = cur[0]
    


def number_of_frames(array_3d):
    #we want to know how many frames a certain file has. 
    #We can access the first dimension of the 2d array vol
    global frame_number
    vol = array_3d[2]
    frame_number = len(vol)



def set_variables(array_3d):
    voltage_channel(array_3d)
    current_channel(array_3d)
    first_voltage_frame(array_3d)
    first_current_frame(array_3d)
    number_of_frames(array_3d)


    

#%%





    
#%% 
         


                   
#%% Holding current, resting membrane potential, input resistance of just one frame
def holding_current_rmp(timeline, cur0, vol0):
    global hold, rmp
    #we define the time interval where we can calculate the rmp and holding current
    interval = np.where( (timeline > 700) & (timeline < 950))
    #we calculate rmp and holding current for the first frame
    rmp = np.mean(vol0[interval])
    hold = np.mean(cur0[interval])
    print("resting membrane potential", rmp, "holding current", hold)


    
def input_resistance(timeline, cur0, vol0):
    global input_res, testcur, testvol, deltavol, deltacur
    #we define the time interval where we can calculate the rmp and holding current
    interval = np.where( (timeline > 700) & (timeline < 950))
    #we calculate rmp and holding current for the first frame
    rmp = np.mean(vol0[interval])
    hold = np.mean(cur0[interval])
    #now we define the time interval where we inject our test current pulse of -25 pA
    test_interval = np.where((timeline > 1100) & (timeline < 1180))
    #get the average current and voltage of this interval
    testcur = np.mean ( cur0 [test_interval] )
    testvol = np.mean ( vol0 [test_interval] )
    #get the difference between holding current / rmp and current and voltage at testpulse
    #since the testcur will always be negative we get the absolute value and add the holding current, which can be poitive or negative to get the absolute difference
    deltacur =  abs(testcur - hold)
     #since the testvol will always be more negative than the rmp7, we can subtract the absolute values to get the difference
    deltavol = abs(testvol) - abs(rmp)
    #to calculate the input resistance we use 
    #R = U / I
    #units Megaohm
    input_res = (deltavol / deltacur)* 1000
    print("input resistance", input_res, "delta U", deltavol, "delta I", deltacur)
   


#%% Ih sag
def Ih_sag(timeline, cur0, vol0, hold, rmp):
    global minV, stabV, Ih_vol
    time_hyp = np.where((timeline > 100) & (timeline < 590))
    time_stabhyp = np.where ((timeline > 350) & (timeline < 590))
    curinj = np.mean(cur0[time_hyp]) - hold #strength of current injection
    minV = np.min(vol0[time_hyp]) - rmp  #maximum hyperopolarization for Ih
    stabV = np.mean(vol0[time_stabhyp]) - rmp # stable hyperpolarization for Ih
    Ih_vol = minV/stabV  
    print("Ih_vol", Ih_vol,"minV", minV, "stabV", stabV)




#%% number of AP per frame and figures
def AP_per_frame_figures (frame_number, vol, cur):
    global cur_list, frame_list, AP_list, peak_list, totalAP_list
    #we will create a list of current injection values in the for loop
    #this list starts at -225 pA and we will add 25pA for every iteration throught the loop
    pA = -225
    #we create empty lists to which we can add values of th efor loops that will follow
    frame_list = list()
    cur_list = list()
    AP_list = list()
    peak_list = list()
    totalAP_list = list()
    total = 0
    for i in range(0, frame_number):
        #for each frame of a file we create a plot with 2 sublplots
        fig01, ax1 = plt.subplots(2)
        fig01.suptitle(i)
        #one subplot ax1 [0] will contain the voltage trace
        #here we use the xtime array because this has the same length as the vol array
        ax1[0].plot(xtime, vol[i])
        ax1[0].set_ylim([-120,50])
        ax1[0].set_xlabel("time [ms]")
        ax1[0].set_ylabel("voltage [mV]")
        #the second subplot ax1[1] contains the current trace
        ax1[1].plot(xtime, cur[i])
        ax1[1].set_ylim([-200,500])
        ax1[1].set_ylabel("current [pA]")
        #find peaks returns ndarray -> indices of peaks that satisfy all given conditions
        peaks, _ = find_peaks(vol[i], height = 0)  
        #we set the count variable to 0 so that we can count the number of APs for each frame while we iterate through the peaks of that frame
        #for every frame this variable will be reset to zero
        count = 0
        #total = total + count
        #now we add 25 for every iteration through the loop
        pA = pA + 25
        for peak in peaks:
            #we add each peak value to a list 
            peak_list.append(peak)
            #we draw a vertical line trhough every peak in the voltage plot
            ax1[0].axvline(x = timeline[peak])
            #for every peak we find in that frame we add 1 to the count variable
            count = count + 1
            #we add 1 to the total count of APS in that recording
            #since this variable is not reset to zero we will get a total number of APs spaning all frames
            total = total + 1
        frame_list.append(i)
        AP_list.append(count)
        cur_list.append(pA)
        totalAP_list.append(total)
        print(i, count, pA) 
    #we create a second plot that shows the injected current on the x axis and the number of APs for each injected current on the y axis
    #this shows the input-output correlation
    fig02, ax2 = plt.subplots()
    ax2.plot(cur_list, AP_list, "bo", color = "red", markersize = 3)
    ax2.set_title("input - output")
    ax2.set_xlabel("injected current [pA]")
    ax2.set_ylabel("number of action potentials")
    
    
    
#%% rheobase
def rheobase (AP_list, cur_list):
    #we iterate through the list of APs per frame
    for i in AP_list:
        if i < 1:
            continue
        #when we find the index of the list where the number of AP is bigger than 1, we save that index/frame as pos
        #then we take the current value of that frame from the cur_list (list of injected current values, 25 pA steps)
        #this current value is the rheobase, the lowest current injection that yields at least 1 AP
        if i >= 1:
            pos = AP_list.index(i)
            rheo = cur_list [pos]
            print("Rheobase:", rheo)   
            break    
        
        
        
        
#%% ISI burstratio
def ISI (AP_list, totalAP_list, peak_list) :
    global burstratio, timeint1, timeint2
    #take the frame that has at least 7 spikes
    for i in AP_list:
        if i < 7:
            frame = "no"
            continue
        if i >= 7:
            frame = AP_list.index(i)
            break
    #now we know with which frame to work
    #to find positions of peaks 1, 2, 6, 7 we have to determine the index of the peaks
    #number of total AP is one bigger than index of list p
    if frame == "no":
        pass
    else:
        p1 = totalAP_list[frame] - AP_list[frame]
        p2 = p1 + 1
        p6 = p1 + 5
        p7 = p1 + 6
        peak1 = peak_list[p1]
        peak2 = peak_list[p2]         
        peak6 = peak_list[p6] 
        peak7 = peak_list[p7]
        isi1 = peak2 - peak1
        isi2 = peak7 - peak6
        #calculate the timeinterval between the first and second spike and the 6th and 7th spike
        timeint1 = isi1 * 0.05
        timeint2 = isi2 * 0.05
        burstratio = timeint2 / timeint1
        print("burstratio", burstratio)
    
#%% decide whether to analyze
#depending on whether our recording has more than 20 frames or not, we will analyse the whole recording or not
def analyse(frame_number):
    if frame_number >= 20:
        holding_current_rmp(timeline, cur0, vol0)
        input_resistance(timeline, cur0, vol0)
        Ih_sag(timeline, cur0, vol0, hold, rmp)
        AP_per_frame_figures (frame_number, vol, cur)  
        rheobase(AP_list, cur_list)
        ISI(AP_list, totalAP_list, peak_list)
    else:
        pass
  


#%%                
#Testfile

#efile1 = "/mnt/live1/jwinterstein/LayerVI_EC/2021/04_2021/130421/Slice1/cellchar_432.cfs"
efile1 = "/mnt/live1/jwinterstein/LayerVI_EC/2021/04_2021/130421/Slice1/cellchar_433.cfs"


read_as_3Darray(efile1)    
create_timeline_array()
create_timeline_plots()
set_variables(array_3d)
analyse(frame_number)