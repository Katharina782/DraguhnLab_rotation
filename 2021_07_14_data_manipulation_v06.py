#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 16:58:58 2021

@author: kmikulik
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pingouin as pg
import math

dir_data = "C:\\Users\\Kathi\\Documents\\Heidelberg\\Epyhs_Labrotation_ChristianThome\\PyCharm Analysis\\Analysis\\"

# when running script on server this is the path to use:
#dir_data = "/mnt/live1/kmikulik/Analysis/"

"""
3 Sinus files containing all recordings of all cells
"""
#pd.read_pickle("")
df_int = pd.read_pickle(dir_data + "pickle_files\\sinus_df_int_new.pkl")
df_int1 = pd.read_pickle(dir_data + "pickle_files\\sinus_df_int_new1.pkl")
df_int2 = pd.read_pickle (dir_data + "pickle_files\\sinus_df_int_new2.pkl")

#remove any values that do not meet the specified conditions
df_int = df_int.query ( "frequency ==  5 or frequency == 20  or frequency == 50 or frequency == 200 or frequency == 500 or frequency == 1000" )

#add all 3 dataframes to one containing all sinus recordings
df_sinus = pd.concat ( [df_int, df_int1, df_int2])
df_sinus.rename ( columns = { "file_num" : "sinus_file"}, inplace = True)



""" spulse dataframe """
df_spulse =  pd.read_pickle(dir_data + "pickle_files\\spulse_all_dataframe.pkl")
df_spulse.sort_values (by = "cell", inplace = True)



"""
read excelsheet with image analysis data
"""

histo_df = pd.read_excel(dir_data + "Histo_data_new.xlsx", sheet_name = "all cells")

#convert cell_id entries to only integers


for i, row in histo_df.iterrows():
    #change entries of cell id so that it fits the other dataframes in the format YYYYMMDDSC 
    #S for slice 
    #C for cell
    cell = row["cell"]
    cell_str = str(cell)
    cell_id = cell_str.replace("_",  "")
    cell_id = cell_id.replace("S", "")
    cell_id = cell_id.replace("C", "")
    histo_df.at[i, "cell"] = cell_id
    
    #calculate the soma area with the formula a * b * pi
    # a equals half the lenght of the soma
    # b equals half the width of the soma
    somadia1 = row["somadia1"]
    somadia2 = row["somadia2"]
    soma_area = (somadia1/2) * (somadia2 /2) * math.pi
    histo_df.at [i, "soma_area"] = soma_area

histo_df.to_excel(dir_data + "final_analysis\\Histo_data_new.xlsx")
#convert dtype to int
histo_df = histo_df.astype({ "cell" : int })



"""
cellchar dataframe
"""

cellchar = pd.read_pickle(dir_data + "pickle_files\\cellchar_dataframe.pkl")
cellchar = cellchar.astype( {"cell_id" : int})
cellchar.rename ( columns = { "cell_id" : "cell", "file" : "cellchar_file" }, inplace = True )


"""combine histo dataframe, cellchar dataframe, 
"""
cell_histo = pd.merge( cellchar, histo_df, how = "inner", on = "cell")

"""
combine sinus dataframe, cellchar dataframe and histo dataframe
"""
#join columns on key "cell" which is the unique id for each cell
master = pd.merge(df_sinus, histo_df, how = "inner", on = "cell")
master_map = pd.merge (master, cellchar, how = "inner", on = "cell")




"""
excel sheet containing all handwritten parameters
"""
cellsheet = pd.read_excel(dir_data + "Cellsheet_handwritten_parameters.xlsx")


#cellsheet.drop(labels = 284, axis = 0, inplace = True)


# in column Virus injection, 1 = injected, 0 equals not injected
for i, row in cellsheet.iterrows():
    date = str( row["Date"] )
    date = date.replace("_", "")
    #print(date)
    sli = str(int(row["Slice"]))
    #print(sli)
    cell = str( int(row["Cell"]) )
    #print(cell)
    cell_id = int(date + sli + cell)
    #print(cell_id)
    cellsheet.at[i, "cell"] = cell_id

cellsheet = cellsheet.astype ( {"cell" : int}) 



cellsheet.drop ( labels = ['Date', 'imaging', "animal", 'sex', 'Slice', 'Cell',
        'pip_ res', 'leak current', 'input_res', 'access_res',
       'Gigaseal_quality', 'RMP', 'bridge', 'RMP_Mkcellch', 'spulse_rmp_1',
       'spulse_rmp_2', 'spulse_rmp_3', 'MKcellchar_70mV', 'spulse_70mV_1',
       'spulse_70mV_2', 'spulse_70mV_3', 'MK_sinus', 'MK_sinus1', 'MK_sinus2',
       'end_inp_res', 'outsideout', 'current_drift', 'comments',
       'Mksinus', '5 Hz (frame)', '20 Hz (frame)', '50 Hz (frame)',
       '200 Hz(frame)', '500 Hz (frame)', '1000 Hz (frame)'], 
                   axis = 1, inplace = True)



#add handwritten parameters to the master_map
master_map = pd.merge(master_map, cellsheet, how = "inner", on = "cell")



"""
The access resistance at the end of the measurement should not be higher than 35 Megaohm
"""
master_map = master_map[master_map["end_acc_res"] < 35]




"""
add a column for AcDnes
# 0 for nonAcD cell
# 1 for AcD cell
"""

for i, row in master_map.iterrows():
    val = row["AcD_stem_length"]
    width = row["AcD_stem_width"]
    if val > 2 :
        master_map.at [i, "AcDnes"] = 1
    else:
        master_map.at [i, "AcDnes"] = 0


"""
add a column for AcDnes as a continuum
"""

for i, row in master_map.iterrows():
    val = row["AcD_stem_length"]
    if val == 0:
        master_map.at [i, "AcDnes_continuum"] = 0
    else:
        master_map.at [i, "AcDnes_continuum"] = val





"""
if the AIS was cut off, AIS length = 0
We set these values to None so it is easier to plot later on
"""
for i, row in master_map.iterrows():
    AIS_length = row[ "AIS_length"]
    if AIS_length == 0:
        master_map.at [i, "AIS_length"] = None   
        
        
        
"""       
for nonAcD -> set AcD_stem_length & AcD_stem_width to None   
"""     
for i, row in master_map.iterrows():
    stem_length = row["AcD_stem_length"]
    stem_width = row["AcD_stem_width"]
    if stem_length == 0:
        master_map.at[i, "AcD_stem_length"] = None
    if stem_width == 0:
        master_map.at[i, "AcD_stem_width"] = None


"""
we cannot simply calculate the median phase per cell with the common median function.
Rather we have to use circular statistics.
the mean is the average direction of a variable in the population
mean resultant vector length is an indication of the spread of the data
"""
from pingouin import convert_angles
from pingouin import circ_rayleigh
import scipy.stats
from scipy.stats import circstd
import math
    
        
#add row with phases in degree + row with phases in radians   
for i, row in master_map.iterrows():
    degree = row["phase"]
    rad = convert_angles(degree, low = 0, high = 360, positive = True)
    master_map.at[i, "radians"] = rad

#name the column that contains phases in degree "degree"        
master_map.rename(columns = {"phase" : "degree"}, inplace = True)



#circular mean with scipy
#mean resultant vector lentgh with pingouin
from scipy.stats import circmean

for i, row in master_map.iterrows():
    cell = row["cell"]
    freq = row["frequency"]
    phase_cell = master_map[ (master_map["cell"] == cell) & (master_map["frequency"] == freq)]["radians"]
    mean = round(circmean(phase_cell), 4)
    resultant_vl = round(pg.circ_r(phase_cell), 4)
    #print(resultant_vl)
    master_map.at[i, "mean"] = mean
    master_map.at[i, "resultant_vl"] = resultant_vl



""" 
remove unwanted columns
"""    
master_map.drop(labels = ["Notes", "numberAP","sinus_file", "frame", "somadia1", "somadia2", "ais_start_x", "ais_start_y", "ais_start_z", 
                    "ais_end_x", "ais_end_y", "ais_end_z", "cell_superficiality",  "cellchar_file", "testcur", "hold", " testvol", "testcur",
                    "deltavol", "deltacur", "minV", "AP_list", "stabV", "interval1", "interval2", "AP_distance", "intercept", "somasize"], 
                   axis = 1, inplace = True)



"""
add a column for the time in ms that has passed since zero degree
"""

for i, row in master_map.iterrows():
    freq = row["frequency"]
    degree = row["degree"]
    #Periodendauer in ms
    period = 1000/freq
    degree_time = period/360
    time_since_0 = degree_time * degree
    master_map.at[i, "time_since_zero"] = time_since_0
    
    


"""
create a dataframe that contains the mean vector direction and length per cell 
"""

master_small = master_map.groupby(["cell", "frequency", "mean", "resultant_vl"]).mean()
#after groupby the columns are now called index
#reset columns
master_small.reset_index(inplace = True)
master_small = master_small.drop(columns = ["degree", "radians"])

master_map.sort_values(by = ["frequency", "cell"], inplace = True)
master_small.sort_values(by = ["frequency", "cell"], inplace = True)
    



"""
for any analysis that does not take into account the phase precision we can make the dataframe even smaller
In master_cell there is only one row for each cell
"""

master_cell = master_small.groupby("cell").mean()
master_cell.reset_index(inplace=True)
master_cell=master_cell.drop(columns=["frequency", "mean", "resultant_vl"])





"""
remove outlier of burstratio
"""

#small dataframe
# outlier_burstratio = list(master_small[ master_small.burstratio == master_small["burstratio"].max()].index)
# master_small.drop (  index = outlier_burstratio, inplace = True)

#big data frame
# outlier_burstratio = list(master_map[ master_map.burstratio == master_map["burstratio"].max()].index)
# master_map = master_map.drop (  index = outlier_burstratio)





"""
save dataframes as excel sheets
"""
master_small.sort_values("cell", inplace = True)

master_small.to_excel(dir_data + "master_map_small.xlsx")


# outlier_burstratio = list(master_map[ master_map.burstratio == master_map["burstratio"].max()].index)
# master_map = master_map.drop (  index = outlier_burstratio)

master_map.to_excel (dir_data + "master_map.xlsx")


master_cell.to_excel(dir_data + "master_cell.xlsx")
