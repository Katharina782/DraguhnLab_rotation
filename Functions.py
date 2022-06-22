#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 14:06:04 2021

@author: kmikulik
"""

import openpyxl

#empty parenthesis indicates that this function does not take any input
def search_validcellchar():
    from openpyxl import load_workbook
    cellsheet = load_workbook("Cellsheet.xlsx")
    #look only at the first sheet in the file called "Patch_data"
    patch = cellsheet["Patch_data"]
    #column 16 contains the cellchar file number
    col16 = patch["P"]
    #creates a list of a ll entries in column 16 -> all file numbers
    cellchar_rec = list()
    for row in range (len(col16)):
        value = col16[row].value
        cellchar_rec.append(value)
    print(cellchar_rec)

    
def cellchar_Ih(x_time, vol, cur):
    #
    timeframe_curinj = np.where ((x_time > 800) & (x_time < 1300))
    timeframe_stabhyp = np.where ((x_time > 1100) & (x_time < 1300))
    curinj = np.mean(cur[timeframe_curinj]) - hold_av # strength of current injection
    minV = np.min(vol[timeframe_curinj]) - rmp #maximum hyperopolarisation for Ih
    stabV = np.mean(vol[timeframe_stabhyp]) - rmp #stable hyperpolarisation for Ih
    Ih_vol = minV/stabV
    
    return [minV, stabV, Ih_vol]


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


       efile1 = f"{kathi}/{date}/{file}"
        
        rec = stfio.read(efile1,"tcfs")
        #convert data into an array
        rec = np.array(rec)
                
        timeline = np.arange(0,2000,0.02).tolist()
        x_time = np.asarray(timeline)
            
        vol = rec[1]
        len_vol = len(vol)
        cur = rec[0]
        len_cur = len(cur)
        for i in range (0, len_vol):
            y_vol = vol[i]
            function()
           
            
        def function (x_time, y_vol)