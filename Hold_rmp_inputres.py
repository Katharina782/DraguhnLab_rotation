#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 16:36:40 2021

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
            
from xlwt import Workbook





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
    timeline = np.arange(0,2000,0.02).tolist()
    timeline = np.asarray(timeline)
    

#%% set_parameters

def voltage_channel(array_3d):
    global vol
    vol = array_3d[1] 
    


def current_channel(array_3d):
    global cur
    cur = array_3d[0]
 


def first_voltage_frame(array_3d):
    global vol0
    vol = array_3d[1]
    vol0 = vol[0]



def first_current_frame(array_3d):
    global cur0
    cur = array_3d[0]
    cur0 = cur[0]
    


def number_of_frames(array_3d):
    global frame_number
    vol = array_3d[1]
    frame_number = len(vol)


def time_intervals():
    #convert time to data points
    global time250, time305, time395
    time250 = int(250/0.02)
    time305 = int(305/0.02)
    time395 = int(395/0.02)

def set_variables(array_3d):
    voltage_channel(array_3d)
    current_channel(array_3d)
    first_voltage_frame(array_3d)
    first_current_frame(array_3d)
    number_of_frames(array_3d)
    time_intervals()
    
def holding_current_rmp(timeline, cur0, vol0):
    global hold, rmp
    #we define the time interval where we can calculate the rmp and holding current
    interval = np.where( (timeline > 0) & (timeline < 250))
    #we calculate rmp and holding current for the first frame
    rmp = np.mean(vol0[interval])
    hold = np.mean(cur0[interval])
    print("resting membrane potential", rmp, "holding current", hold)


    
def input_resistance(timeline, cur0, vol0):
    global input_res, testcur, testvol, deltavol, deltacur
    #we define the time interval where we can calculate the rmp and holding current
    interval = np.where( (timeline > 0) & (timeline < 250))
    #we calculate rmp and holding current for the first frame
    rmp = np.mean(vol0[interval])
    hold = np.mean(cur0[interval])
    #now we define the time interval where we inject our test current pulse of -25 pA
    test_interval = np.where((timeline > 305) & (timeline < 395))
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
    

    




#%% analysis
def analyse(frame_number):
    if frame_number >= 20:
        av_hold_rmp_inputres(frame_number, vol, cur)
        hold_rmp_inputres(cur0, vol0, vol)
        # Ih_sag(timeline, cur0, vol0, hold, rmp)
        # AP_per_frame_figures (frame_number, vol, cur)  
        # rheobase(AP_list, cur_list)
        # ISI(AP_list, totalAP_list, peak_list)
        # ap_analysis()
        
        
        
#%%                
#Testfile
#efile1 = "mnt/live1/cthome/CThome_ProjectArchive/Y_AISdiversity_PYR/MikulikKatharina_Data/KM_Data/23_02_2021"
#efile1 = "/mnt/live1/kmikulik/recordings/23_02_2021/MKcellchar_021.cfs"
#efile1 = "/mnt/live1/jwinterstein/LayerVI_EC/2021/04_2021/130421/Slice1/cellchar_432.cfs"
#efile1 = "/mnt/live1/kmikulik/recordings/01_03_2021/MKcellchar_034.cfs" 
efile1 = "/mnt/live1/kmikulik/recordings/test/2021_03_29/MKcellchar_225.cfs"
#efile1 = "/mnt/live1/kmikulik/recordings/23_02_2021/MKcellchar021_cfs"
read_as_3Darray(efile1)    
create_timeline_array()
set_variables(array_3d)
analyse(frame_number)
        
