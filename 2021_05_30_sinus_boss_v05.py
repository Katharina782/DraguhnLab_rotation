#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  9 19:45:48 2021

@author: kmikulik
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 08:18:59 2021

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






#the sinus protocol has a sampling rate of 5000/sec
#each frame lasts for 2sec
#there are 2 channels: current and voltage
#file = '/mnt/live1/kmikulik/recordings/2021_03_06/MKsinus_006.cfs'
# #file = '/mnt/live1/kmikulik/recordings/2021_03_06/MKsinus_008.cfs'
# file = '/mnt/live1/kmikulik/recordings/2021_03_06/MKsinus_007.cfs'
# #read file with stfio library   
# rec = stfio.read(file, "tcfs")
# #create a 3d array of the recording file
# rec = np.array(rec)

# """
# DEFINE CONSTANTS
# """
# TIMELINE = np.arange(0,2000, 0.02)
# #create a timeline for the 2 sec sweeps of the recording
# #zero to 2000 ms, with 0.02 ms steps inbetween
# x_time = np.arange(0,2000,0.02)


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

#%%
"""
DEFINE CONSTANTS
"""
timeline = np.arange(0,2000, 0.02)




#%%
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
def find_phase (rec, frame_thr, freq_frame):
    #time_180 = start_period(rec, frame)
    phase_test = {}
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
            degree_list.append(degree)
        phase_test[frame] = degree_list
    return phase_test
        


                              

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

    






def analyse_file(rec, timeline):
    frames_for_analysis = discard_frames(rec)
    freq_frame = frequency_per_frame(rec, frames_for_analysis, timeline)
    #period = period(freq_frame)
    peak_frame = APs_per_frame (rec, frames_for_analysis)
    ap_vol, ap_time = find_AP_intervals(rec, peak_frame)
    frame_thr = find_thresholds_per_frame(ap_vol, ap_time)
    #letzte Zeile könnte Bug sein
    #phase_thr = get_phase_for_APs(frame_thr)
    phase_test = find_phase(rec, frame_thr, freq_frame)
    return peak_frame, frames_for_analysis, freq_frame, frame_thr, phase_test
    


  



#frame_list, freq_list, AP_count_list, phase_list, cell_total, file_list


""" 
In an excel sheet we have all the cellchar files we want to analyse for a certain cell.
In our folders we have many additional files, some of which we do not need to analyse.
Using the excelsheet we get a list of all filenames we want to analyse.
"""

def files_for_analysis(cellsheet):
    # save excelsheet as datafram
    cellsheet_df = pd.read_excel(cellsheet)
    # get the 3 columns that contains the cellchar number and convert entries to list
    sinus_for_analysis = cellsheet_df.MK_sinus [15:284].tolist()
    #sinus_for_analysis = [x for x in column1 if str(x) != 'nan']
    excel_filename_list = list()
    for i in sinus_for_analysis:
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
        excel_filename = f"MKsinus_{i}.cfs"
        excel_filename_list.append(excel_filename)
    return sinus_for_analysis, excel_filename_list


"""
We want to create a unique id for each cell that we recorded and stained, so we can combine recording data and histology later on.
The unique id will contain Slice number and Cell number "SxCy"
In the main function we will also add the date "YYYY_MM_DD_SxCy"
"""
def create_cell_id(cellsheet):
    # save excelsheet as datafram
    cellsheet_df = pd.read_excel(cellsheet)
    #use function to get list of all sinusfile numbers
    sinus, excel_filename_list = files_for_analysis(cellsheet)
    # get the columns that contain Slice and Cell number
    slice_num = cellsheet_df.Slice[15:284].tolist()
    cell_num = cellsheet_df.Cell[15:284].tolist() 
    # create empty list for cell id
    cell_dict = {}
    # iterate through both lists and get the slice and cell number, but only if the entry at index i of the cellchar_70mV list is not "nan"
    for i in range(len(sinus)):
        no_entry = np.isnan(sinus)

        if no_entry[i] == True:
            continue

        else:
            file_num = int( sinus[i])
            cell = str(cell_num[i])
            sli = str(slice_num[i])
            cell_id = f"{sli}{cell}"
            cell_dict[file_num] = cell_id
    return cell_dict

        
"""
 In some cases the AP are not analysable. If that is the case they are set to "undefined"
 We cannot plot a dataframe that contains values which are not defined.
 Therefore we filter the column "phase" for undefined values and remove them
"""
def filter_dataframe(dataframe):
    df_defined = dataframe.query ( 'phase != "undefined"')
    df_int = df_defined.astype({ "phase" : int, "cell" : int })
    return df_int      
        
        
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
create a dataframe with freq, frame, AP count, phases
"""
def create_dataframe_entries(rec, timeline, date, file, freq_list, AP_count_list, phase_list, frame_list, cell_dict, cell_total, file_list):
    peak_frame, frames_for_analysis, freq_frame, frame_thr, phase_test = analyse_file(rec, timeline)
    for frame, phases in phase_test.items():
        #freq = freq_frame[frame]
        #freq_list.append(freq)
        AP_count = 0
        for phase in range(0, len(phases)):
            freq = freq_frame[frame]
            freq_list.append(freq)
            AP_count += 1
            AP_count_list.append(AP_count)
            phase = phase_test[frame][phase]
            phase_list.append(phase)
            frame_list.append(frame)
            day = date.replace("_", "")
            #create a unique cell_id by adding date and slice and cell number (from cell_list) together into one string
            file_num = file.replace("MKsinus_", "")
            file_num = int( file_num.replace(".cfs", "") )
            cell_id = f"{day}{cell_dict[file_num]}"
            cell_total.append(cell_id)
            file_list.append( file_num )

    return frame_list, freq_list, AP_count_list, phase_list, cell_total, file_list


"""
Here you can add the cellsheet you want to analyse and the directory where your files are located
Then the main function executes the analysis for all your files in that directory.
"""

# define which excel sheet to use
# it should contain the numbers of cellchar recordings
# /mnt/live1/kmikulik/Analysis/Cellsheet_Patcherei.xlsx"
cellsheet = "/mnt/live1/kmikulik/Analysis/Cellsheet_handwritten_parameters.xlsx"

# set the directory where your files for analysis are located
directory = "/mnt/live1/kmikulik/recordings/Analysis/"
#directory = "/mnt/live1/kmikulik/recordings/test/"




def main(cellsheet, directory):
    """
    This funtion iterates through all files of interest and performs the analysis of these files.
    At the end it adds the output of each iteration to the corresponding row of the excel sheet we create at the beginning.
    """

    # create empty lists to which we will add the values of the analysis
    freq_list = []
    AP_count_list = []
    phase_list =[]
    frame_list = []
    cell_total = []
    file_list = []

    # first we create a list of files we want to analyse
    # the input of this function is an excel sheet
    sinus_for_analysis, excel_filename_list = files_for_analysis(cellsheet)
    #then we create a list of cell and slice numbers for each file we want to analyse 
    #this funciton returns a list with entries "SxCy"
    cell_dict = create_cell_id(cellsheet)
    
    # creaete a dictionary of all files in the specified directory with dates (folders) as keys
    dictionary = file_dictionary(directory)
    for date, files in dictionary.items():
        files_oi = []
        # loop through all files of each date and filter for files with the correct names: MKcellchar_xxx.cfs
        for file in files: 
            if re.search("sinus.*.cfs", file) != None:
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
            file_name = f"{directory}/{date}/{file}"
            
            #Now that we have only one single file specified by file_name, we can start the analysis of that file:
            rec = read_file(file_name)
            timeline = create_timeline_array()
            peak_frame, frames_for_analysis, freq_frame, frame_thr, phase_test = analyse_file(rec, timeline)
            # set_variables(array_3d)
            # the function analyse_cellchar returns a tuple with all calculated values
            frame_list, freq_list, AP_count_list, phase_list, cell_total, file_list = create_dataframe_entries(rec, timeline, date, 
                                                                                                    file, freq_list, AP_count_list, 
                                                                                                    phase_list, frame_list, cell_dict,
                                                                                                    cell_total, file_list)



    df_raw = pd.DataFrame (data = {"cell" : cell_total, "file_num" : file_list, "frame" : frame_list, "frequency" : freq_list, 
                                  "numberAP" : AP_count_list, "phase" : phase_list}, dtype = float)   
    #with this function we can remove any "undefined" values    
    df_int = filter_dataframe (df_raw)
    
    return df_raw, df_int

            
df_raw2, df_int2 = main(cellsheet, directory)           

# df_raw2.to_pickle("/mnt/live1/kmikulik/Analysis/sinus_df_raw_new2.pkl")
# df_int2.to_pickle("/mnt/live1/kmikulik/Analysis/sinus_df_int_new2.pkl")


"""
Create plots to visualize the data
In some cases the AP are not analysable. If that is the case they are set to "undefined"
We cannot plot a dataframe that contains values which are not defined.
Therefore we filter the column "phase" for undefined values and remove them
"""
# av_phase = df_int.groupby ( ["frequency", "cell"], as_index = False) ["phase"].mean()
# av_phase = df_int.groupby ( "frequency", as_index = False) ["phase"].mean()
# #av, ax = plt.subplots()
# sns.catplot(x = "frequency", y = "phase", kind = "box", data = av_phase )
# #sns.stripplot(x = "frequency", y = "phase", data = av_phase, color = "black" , size = 7)

# #compare standard deviations with original data
# raw, ax = plt.subplots()
# sns.catplot ( x = "frequency", y = "phase", kind = "box", data = df_int)
# sns.stripplot

# std_phase = df_int.groupby ( [ "frequency", "cell" ] , as_index = False ) ["phase"].std()
# std, ax = plt.subplots()
# sns.catplot ( x = "frequency", y = "phase", kind = "box", data = std_phase )
# sns.stripplot ( x = "frequency", y = "phase", data = std_phase, color = "black", size = 7)



# df_raw.to_pickle("/mnt/live1/kmikulik/Analysis/sinus_df_raw2.pkl")

# d = pd.read_pickle("/mnt/live1/kmikulik/Analysis/sinus_df_raw.pkl")

# df_int.to_pickle("/mnt/live1/kmikulik/Analysis/sinus_df_int2.pkl")

# di = pd.read_pickle("/mnt/live1/kmikulik/Analysis/sinus_df_int.pkl")


# di_filt = di.query ( "frequency ==  5 or frequency == 20  or frequency == 50 or frequency == 200 or frequency == 500 or frequency == 1000" )
                    
# raw, ax = plt.subplots()
# sns.catplot ( x = "frequency", y = "phase", kind = "box", data = di_filt)
# sns.stripplot ( "frequency", y = "phase", data = di_filt, color = "black", size = 3)           

           