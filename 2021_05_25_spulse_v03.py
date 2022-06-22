#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 16:58:31 2021

@author: kmikulik
"""

import numpy as np
import pandas as pd
import seaborn as sns
sns.set(style="ticks", palette='rocket')

import matplotlib.pyplot as plt
from matplotlib import rc
rc("pdf", fonttype=42)
from matplotlib.backends.backend_pdf import PdfPages

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


          

"""
the spulse file can be read as a 3d array
the 3 dimensions of this array are:
    1. 2 channels -> voltage and current
    2. a varying number of frames
    3. 25 000 datapoints in 0.5 sec
"""

def read_as_3Darray(file_name):
    rec = stfio.read(file_name,"tcfs")
    #convert data into an array
    array_3d = np.array(rec)
    #np.arrange creates arrays with regularly incrementing values
    #first value is start of array, second is end of array
    #in this case 0.02 will be the increment
    return array_3d
    
""" one frame has 500 ms, intervals are 0.02ms """
def create_timeline_array():       
    timeline = np.arange(0,500,0.02).tolist()
    timeline = np.asarray(timeline)
    return timeline
    

#%% set_parameters

def voltage_channel(array_3d):
    vol = array_3d[1] 
    return vol
    


def current_channel(array_3d):
    cur = array_3d[0]
    return cur
 


def number_of_frames(array_3d):
    vol = array_3d[1]
    frame_number = len(vol)
    return frame_number


    
    
def create_figures(array_3d, timeline):   
    frame_number = number_of_frames (array_3d)
    cur = current_channel (array_3d)
    vol = voltage_channel (array_3d)
    
    for i in range(0, frame_number):
        #for each frame in loop create a plot with 2 subplots (one for current and one for voltage)
        fig01, ax1 = plt.subplots(2)
        #the title of the plot should be the frame number i
        fig01.suptitle(i)  
        #the first plot is voltage vs. time
        #set time interval for plotting current and voltage, where we inject our current pulse of 10ms
        interval = np.where ((timeline > 140 ) & ( timeline <180))
        ax1[0].plot(timeline[interval], vol[i][interval])
        ax1[0].set_ylim([-100,70])
        ax1[0].set_xlabel("time [ms]")
        ax1[0].set_ylabel("voltage [mV]")
        #draw a vertical line at the end of the current injection to be able to see whether the AP threshold is inside of the current pulse
        ax1[0].axvline(x = 160, color = "r")
        #second plot is current vs. time
        ax1[1].plot(timeline[interval], cur[i][interval])
        ax1[1].set_ylim([-200, 600])
        ax1[1].set_ylabel("current [pA]")
        ax1[1].set_xlabel("time [ms]")
        #find all peaks in the voltage channel
        peaks, _ = find_peaks(vol[i], height = 0)
        #for each peak add a vertical line
        #for peak in peaks:
            #ax1[0].axvline(x = timeline[peak])
        # pdf_directory = "/mnt/live1/kmikulik/recordings/test"
        # spulse = "/spulse_"
        # frame_num = "frame"
        # file_ending = ".png"
        # pdf_name = f"{pdf_directory}/{date}{spulse}{file}_{frame_num}{i}_{file_ending}"
        # #pp = PdfPages(pdf_name)
        # #pp.savefig(fig01)
        # plt.savefig(pdf_name)
        
                      
            

#%%
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
    x_time_ny  = np.linspace(x_time[0], x_time[-1], len(y_voltage_ny))

    y_voltage_ny = y_voltage_ny[99:-100:1]
    x_time_ny= x_time_ny[99:-100:1]
    
    return [x_time_ny,y_voltage_ny]

def determine_threshold(x_time, y_voltage):
    """ determine the threshold of the AP. """
    thr_section = np.nan
    x_time_ny, y_voltage_ny = AP_upsample(x_time, y_voltage)
    dat_ds = np.diff(y_voltage_ny)/np.median(np.diff(x_time_ny))
    #indices where voltage slope > 20
    ind_thr = np.nonzero(dat_ds > 20)[0]
    #
    if len(ind_thr) > 0:
        thr_section = y_voltage_ny[ind_thr[0]-1]
        ind_thr = ind_thr[0]-1
    
    return [thr_section, ind_thr]
#this function determines threshold potential




#%% first AP in 10ms
"""
Find the first AP of a recording with a threshold within the first 10 ms
this will be the AP to analyse
this will also give you the threshold of this AP
also it will give you y_voltage = vol_int and x_time = time_int for other functions of AP analysis
"""

def find_first_ap(timeline, array_3d):
    # to detrmine the first AP with a threshold within the 10ms current injection interval (150 to 160ms)
    #create a list that contains each framenumber of this file as an entry
    vol = voltage_channel(array_3d)

    peakpos = {}
    AP_list = []
    distance_dict = {}
    
    #create a list that contains each framenumber of this file as an entry
    frame_list = np.arange(0, len(vol), 1).tolist()

    for frame in frame_list:
        peaks, peakind = find_peaks(vol[frame], height=0)
        peakpos[frame] = peaks
        count = 0
        #create a list with length frame_number and each index contains the number of peaks (0 or 1)
        for peak in peaks:
            count += 1
        AP_list.append (count)
    #define voltage and time interval of the first AP for each frame
    # for this we need to determine the threshold for all APs!
    first_ap_vol = {}
    first_ap_time = {}
    for frame in frame_list:
        #if the AP-list contains a peak the value will be one, so we want to get this peak
        if AP_list[frame] > 0:
            # we get voltage values around the first peaks (-100 to +150 data points from peak)
            # we can access the first peak via our dictionary frames
            vol_first_peak = vol[frame][(peakpos[frame][0]-100):(peakpos[frame][0]+150)]
            time_first_peak = timeline[(peakpos[frame][0]-100):(peakpos[frame][0]+150)]
            first_ap_vol[frame] = vol_first_peak
            first_ap_time[frame] = time_first_peak
        else:
            continue
        
    #determine AP threshold for the first AP in each frame
    for frame in frame_list:
        if AP_list[frame] > 0:
            #set variable names for upsampling funciton
            y_voltage = first_ap_vol[frame]
            x_time = first_ap_time[frame]
            thr_section = np.nan
            #now we want to upsample the voltage and time interval around the first peak 
            #(this interval was defined in the previous for loop)
            x_time_ny, y_voltage_ny = AP_upsample(x_time, y_voltage)
            #calculate voltage change / time (dV/ds)
            # first derivative of voltage values in that interval
            dat_ds = np.diff (y_voltage_ny) / np.median (np.diff (x_time_ny) ) 
            #indices where voltage slope > 20
            ind_thr = np.nonzero(dat_ds > 20)[0]
    #this gives you the threshold of this AP at the same time
            if len(ind_thr) > 0:
                thr_vol = y_voltage_ny[ind_thr [0] - 1]
                ind_thr = ind_thr [0] - 1
                thr_time = x_time_ny [ind_thr]
                #calculate the distance between the end of the current injection  at 160 ms 
                #and the time of the AP threshold
                distance = 160 - thr_time 
                if distance > 0:
                #add this distance to the corresponding frame key of a dictionary
                    distance_dict [frame] = distance
    if len( distance_dict ) > 0:         
    #find the first peak with a threshold within the current injection interval (150 to 160 ms)
        AP_distance = min(distance_dict.items(), key = lambda x : x[1]) [1] #just get the value of the distance, not the key frame
        frame_ap_analysis = min(distance_dict.items(), key = lambda x : x[1]) [0] # just the key and not the distance value
        ap_for_analysis = peakpos [frame_ap_analysis] [0]
        y_vol = vol [frame_ap_analysis] [(peakpos[frame_ap_analysis] [0] - 100) : (peakpos [frame_ap_analysis] [0] + 150)]
        x_tim = timeline [ (peakpos [frame_ap_analysis] [0] - 100) : (peakpos [frame_ap_analysis] [0] + 150)]     
    else:
        ap_for_analysis = "undefined"
        frame_ap_analysis = "undefined"
        thr_vol = "undefined"
        thr_vol = "undefined"
        y_vol = "undefined"
        x_tim = "undefined"
        AP_distance = "undefined"
    return ap_for_analysis, AP_distance, frame_ap_analysis, thr_vol, y_vol, x_tim  
            

    
    

#%%rapidness
#%% calculate rapidness and plot rapidness
def determine_rapidness(x_tim, y_vol, thr_vol, date, file):
    """ determine the onset rapidness of the AP. """
    slope = np.nan
    #x_time, y_voltage = AP_upsample(x_time, y_voltage)
    #dat_ds = slope (mv/mS)
    dat_ds = np.diff(y_vol)/np.median( np.diff(x_tim) )

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
    slope, intercept = np.polyfit(y_vol[indrap],dat_ds[indrap],1)
    #this  gives us the slope and y intercept of a linear function
    # determines voltage range for plotting rapidness -> 3 voltage values around the threshold voltage
    thresllist = [thr_vol -2, 
                  thr_vol,
                  thr_vol +5]
    #list of 3 voltage values
    APrapidness = [slope, intercept]
    #abline values = 3 
    abline_values = [APrapidness[0] * i + APrapidness[1] for i in thresllist]
    
    fig02, axs = plt.subplots(2,1, figsize=(6,6), sharex=False) # sharey = sharing of y axis beschriftung
    fig02.suptitle('Action potential')
    
    axs[0].plot(x_tim, y_vol)
    axs[0].axhline(thr_vol, color='r', linestyle='-')
    axs[0].set_xlabel('time (ms)')
    axs[0].set_ylabel('voltage (mV)')
    #[0:-1], geht bis zum vorletzten Wert
    axs[1].plot(y_vol[0:-1], dat_ds)
    axs[1].axhline(y=20, color='r', linestyle='-')
    axs[1].plot(thresllist, abline_values, 'b')
    axs[1].set_xlabel('voltage (mV)')
    axs[1].set_ylabel('slope (mV/ms)')
    #print("rapidness", slope, intercept)
    # save the figure as a pdf in the specified directory
    pdf_name = f"/mnt/live1/kmikulik/recordings/Analysis/{date}/{file}_AP_rapidness_spulse.png"
    #pp = PdfPages(pdf_name)
    #pp.savefig(fig01)
    plt.savefig(pdf_name)
    return slope, intercept
    
    
    
#%% amplitude
def determine_amplitude(x_tim, y_vol, thr_vol):
    """ determine the amplitude of the AP. """
    APamp = np.nan
    #x_time, y_voltage = AP_upsample(x_time, y_voltage)
    low = thr_vol
    peakind, peaks= find_peaks(y_vol,height=0)
    #das zieht nur den Wert des peaks aus der Liste, weil die Liste nur einen value hat -> ersten index [0] 
    up = peaks['peak_heights'][0]
    APamp = up-low
    print("amp", APamp)
    return APamp




#%%halfwidth
def determine_halfwidth(x_tim, y_vol, thr_vol):
    """ determine the halfwidth of the AP. """
    APamp = determine_amplitude(x_tim, y_vol, thr_vol)
    APhw = np.nan
    #if we divide the amplitude by 2, we have half the distance between threshold and peak.
    #if we add this distance to the threshold, we get the voltage value at this half height
    hAPamp = thr_vol + (APamp/2)
    #np.where will print an array containing a list of all indices at which y-voltage satisfies the condition
    #with the first [0] we get the list at the first index of the array 
    #with the second [0] we get the first index value of the list -> this is the point on the left flank of the cure
    firsthind = np.where(y_vol >= hAPamp)[0][0]
    #by taking the last value of the list, we get the point on the right flank of the curve
    lasthind = np.where(y_vol >= hAPamp)[0][-1]
    #now we subtract the time values at the left flank point and the right flank point in order to get the halfwidth
    APhw = x_tim[lasthind]-x_tim[firsthind]
    #print("halfwidth", APhw)
    return APhw



#%% AP rise time
def determine_APriseT(x_tim, y_vol, thr_vol):
    """ determine the halfwidth of the AP. """
    APamp = determine_amplitude(x_tim, y_vol, thr_vol)
    APriseT = np.nan
    #x_time, y_voltage = AP_upsample(x_time, y_voltage)
    AP10perc = thr_vol + (0.1*APamp)
    AP90perc = thr_vol + (0.9*APamp)
    firsthind = np.where(y_vol >= AP10perc)[0][0]
    lasthind = np.where(y_vol >= AP90perc)[0][0]
    APriseT = x_tim[lasthind]-x_tim[firsthind]  
    #print("APrisetime", APriseT)
    return APriseT



#%% AP fall time
def determine_APfallT(x_tim, y_vol, thr_vol):
    """ determine the halfwidth of the AP. """
    APamp = determine_amplitude(x_tim, y_vol, thr_vol)
    APfallT = np.nan
    #x_time, y_voltage = AP_upsample(x_time, y_voltage)
    AP10perc = thr_vol + (0.1*APamp)
    AP90perc = thr_vol + (0.9*APamp)
    firsthind = np.where(y_vol >= AP90perc)[0][-1]
    lasthind = np.where(y_vol >= AP10perc)[0][-1]
    APfallT = x_tim[lasthind]-x_tim[firsthind]  
    #print("APfalltiem", APfallT)
    return APfallT



#%% max AP  rise
def determine_APmaxrise(x_tim, y_vol):
    """ determine the halfwidth of the AP. """
    APmaxrise = np.nan
    #x_time, y_voltage = AP_upsample(x_time, y_voltage)
    dat_ds = np.diff(y_vol)/np.median( np.diff(x_tim) )
    APmaxrise = dat_ds.max()
    print("max AP rise", APmaxrise)
    return APmaxrise
    
    

#%%
#%% AP analysis
def ap_analysis (timeline, array_3d, date, file):
    ap_for_analysis, AP_distance, frame_ap_analysis, thr_vol, y_vol, x_tim = find_first_ap (timeline, array_3d)
    #create_figures(frame_number, cur, vol)

    #in some files you will not be able to analyse the AP, because the measurement was not good:
    if ap_for_analysis == "undefined" :
        slope = "undefined"
        intercept = "undefined"
        APamp = "undefined"
        APriseT = "undefined"
        APfallT = "undefined"
        APmaxrise = "undefined"
        APhw = "undefined"
            
    else:
        slope, intercept = determine_rapidness(x_tim, y_vol, thr_vol, date, file)
        APamp = determine_amplitude(x_tim, y_vol, thr_vol)
        APhw = determine_halfwidth(x_tim, y_vol, thr_vol)
        APriseT = determine_APriseT(x_tim, y_vol, thr_vol)
        APfallT = determine_APfallT(x_tim, y_vol, thr_vol)
        APmaxrise = determine_APmaxrise(x_tim, y_vol)
    return  (thr_vol, AP_distance,  slope, intercept, APamp, APhw, APriseT, APfallT, APmaxrise)





#%%                

efile1 = "/mnt/live1/kmikulik/recordings/test/2021_03_29/MKspulse_631.cfs"

# read_as_3Darray(efile1)    
# create_timeline_array()
# set_variables(array_3d)
# create_figures(frame_number, cur, vol)
# ap_analysis()



#%%



""" 
In an excel sheet we have all the cellchar files we want to analyse for a certain cell.
In our folders we have many additional files, some of which we do not need to analyse.
Using the excelsheet we get a list of all filenames we want to analyse.
"""


def files_for_analysis(cellsheet):
    #save excelsheet as datafram
    cellsheet_df = pd.read_excel(cellsheet)
    #get the column that contains the cellchar number
    #get all spulse numbers from the 3 columns and put them into one list
    spulse_for_analysis = cellsheet_df.spulse_70mV_3 [35:284].tolist()
    #spulse_70mV_2 = list(cellsheet_df.spulse_70mV_2[53:224])
    #spulse_70mV_3 = list(cellsheet_df.spulse_70mV_3[53:224])
    #spulse_for_analysis = spulse_70mV_1 + spulse_70mV_2 + spulse_70mV_3       
    
    #get the files you want to analyse
    spulse = "MKspulse_"
    cfs = ".cfs"
    excel_filename_list = list()
    for i in spulse_for_analysis:
        #if not (pd.isna(i)):
        #    i = int(i)
        i = str(i)
        i = str.split(i, ".")[0]
        try:
            i = int(i)
            i = "{:0>3d}".format(i)
        except:
            pass
        excel_filename = f"{spulse}{i}{cfs}"
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
    spulse_for_analysis = cellsheet_df.spulse_70mV_3 [35:284].tolist()
    # get the columns that contain Slice and Cell number
    slice_num = cellsheet_df.Slice[35:284].tolist()
    cell_num = cellsheet_df.Cell[35:284].tolist()
    # create empty list for cell id
    cell_dict = {}
    # iterate through both lists and get the slice and cell number, but only if the entry at index i of the cellchar_70mV list is not "nan"
    for i in range ( len(spulse_for_analysis) ):
        no_entry = np.isnan(spulse_for_analysis)

        if no_entry[i] == True:
            continue

        else:
            file_num = int( spulse_for_analysis [i])
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

          

"""
Here you can add the cellsheet you want to analyse and the directory where your files are located
Then the main function executes the analysis for all your files in that directory.
"""


# define which excel sheet to use
# it should contain the numbers of cellchar recordings
# /mnt/live1/kmikulik/Analysis/Cellsheet_Patcherei.xlsx"
CELLSHEET = "/mnt/live1/kmikulik/Analysis/Cellsheet_Patcherei_ohne hash.xlsx"
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
            if re.search("spulse.*.cfs", file) != None:
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
            value_tuple = ap_analysis(timeline, array_3d, date, file)
                
            #after analysing we can now add all values to a dictionary (filenames will be the keys of the dictionary)
            #make a list that contains all values of a certain file
            value_list = list(value_tuple)
            #create a unique cell_id by adding date and slice and cell number (from cell_list) together into one string
            
            cell_id, file_num = get_cell_id_for_file(file, date, cell_dict)

            value_list.insert(0, cell_id)
            value_list.insert(1, file_num)
            # for each file (key of dictionary) we add the list as values to a dictionary
            for value in value_tuple:
                value_dict[file] = value_list
                
    dataframe = pd.DataFrame.from_dict(data=value_dict, orient="index", 
                                       columns=["cell_id", "file_num", "thr_vol", "AP_distance", "slope", "intercept",
                                                 "APamp", "APhw",                                    
                                                 "APriseT", "APfallT", "APmaxrise"])      
    #We do not want the file_names as index
    dataframe.index.name = 'File'
    dataframe.reset_index (inplace = True)
    return dataframe                            

dataframe = main (CELLSHEET, DIRECTORY)

# df_spulse1 = dataframe.drop(["File"], axis = "columns")
# df_spulse1.to_pickle("/mnt/live1/kmikulik/Analysis/spulse1_dataframe.pkl")

# df_spulse2 = dataframe.drop(["File"], axis = "columns")
# df_spulse2.to_pickle("/mnt/live1/kmikulik/Analysis/spulse2_dataframe.pkl")

df_spulse3 = dataframe.drop(["File"], axis = "columns")
df_spulse3.to_pickle("/mnt/live1/kmikulik/Analysis/spulse3_dataframe.pkl")



spulse1 = pd.read_pickle("/mnt/live1/kmikulik/Analysis/spulse1_dataframe.pkl")
spulse2 = pd.read_pickle("/mnt/live1/kmikulik/Analysis/spulse2_dataframe.pkl")
spulse3 = pd.read_pickle("/mnt/live1/kmikulik/Analysis/spulse3_dataframe.pkl")

df_spulse_raw = pd.concat( [ spulse1, spulse2, spulse3 ] )



def filter_dataframe(df_spulse_raw):
    df_defined = df_spulse_raw.query ( 'thr_vol != "undefined"')
    df_spulse = df_defined.astype({ "cell_id" : int, "thr_vol" : float, "AP_distance" : float, "slope" : float, "intercept" : float, "APamp" : float, "APriseT": float, "APmaxrise" : float })
    df_spulse.rename ( columns = { "cell_id" : "cell", "file_num" : "spulse_file"}, inplace = True) 
    return df_spulse   

df_spulse = filter_dataframe(df_spulse_raw)

df_spulse.to_pickle("/mnt/live1/kmikulik/Analysis/spulse_all_dataframe.pkl")
df_spulse =  pd.read_pickle("/mnt/live1/kmikulik/Analysis/spulse_all_dataframe.pkl")

#df_spulse = df_spulse.astype( {"cell_id": int, "thr_vol" : int} )

df_spulse_av = df_spulse.groupby( ["cell"], as_index = False) ["thr_vol", "slope", "intercept", 
                                                                  "APamp", "APhw", "APriseT","APfallT", "APmaxrise"].mean()
df_spulse_av = df_spulse.groupby( ["cell_id"], as_index = False) ["thr_vol"].mean()
