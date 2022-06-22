#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 08:14:00 2021

@author: kmikulik
"""


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pingouin as pg
import math
sns.set()



"""
cellchar dataframe
"""
cellchar = pd.read_pickle("/mnt/live1/kmikulik/Analysis/pickle_files/cellchar_dataframe.pkl")
cellchar = cellchar.astype( {"cell_id" : int})
cellchar.rename ( columns = { "cell_id" : "cell", "file" : "cellchar_file" }, inplace = True )
cellchar.sort_values(by = "cell")



""" spulse dataframe """
df_spulse =  pd.read_pickle("/mnt/live1/kmikulik/Analysis/pickle_files/spulse_all_dataframe.pkl")
df_spulse.sort_values (by = "cell", inplace = True)
df_spulse = df_spulse.astype({"APhw": float, "APfallT":float})

#calculate the mean values over all 3 recordings
mean_spulse = df_spulse.groupby ("cell").mean()
mean_spulse.reset_index(inplace = True)



"""
cellchar and sinus and histo data combined
"""
master_map = pd.read_excel ("/mnt/live1/kmikulik/Analysis/master_map.xlsx")
master_small = pd.read_excel("/mnt/live1/kmikulik/Analysis/master_map_small.xlsx")
cellchar_df = master_small.groupby(by = "cell").mean()
cellchar_df.drop(labels = ['Unnamed: 0', "frequency", "mean", "resultant_vl",  'Unnamed: 39'], axis = 1, inplace = True)
cellchar_df.reset_index(inplace = True)
cellchar_df.sort_values(by = "cell")



#this is a messy data set, because we have repeated-measures -> eeach row corresponds to the unit of data collection = "cell-id"
comparison = pd.merge( cellchar_df, mean_spulse, how = "inner", on = "cell")
# in this dataframe x indicates values from cellchar recordings and y indicates values from spulse recordings


        

        


cells_for_analysis = cellchar_df["cell"].to_list()

drop_list = []
for i in mean_spulse["cell"]:
    if i in cells_for_analysis:
        continue
    else:
        drop_list.append(i)
        
#remove all values which are not in the cellchar file:
df_spulse =  pd.read_pickle("/mnt/live1/kmikulik/Analysis/pickle_files/spulse_all_dataframe.pkl")
df_spulse.sort_values (by = "cell", inplace = True)
df_spulse = df_spulse.astype({"APhw": float, "APfallT":float})
mean_spulse = df_spulse.groupby ("cell").mean()


mean_spulse_short = mean_spulse.drop(labels = drop_list, axis = 0)
mean_spulse_short.reset_index(inplace = True)

#concatenate the spulse files and the cellchar files
comparison_filt = pd.concat([mean_spulse_short, cellchar_df], keys = ["spulse", "cellchar"], join = "inner")
comparison_filt.sort_values(by = "cell")
comparison_filt.reset_index(inplace = True)
comparison_filt.rename(columns = {"level_0":"recording_protocol"},inplace = True)
        




"""
visualization
"""
fig, axes = plt.subplots(1, 7, figsize=(40, 5), sharey=False)
fig.suptitle("cellchar n = 61, spulse n = 47")
sns.boxplot(ax = axes[0], data = comparison_filt, y = "APamp", x = "recording_protocol")
sns.swarmplot(ax = axes[0], data = comparison_filt, y = "APamp", x = "recording_protocol", color = "black")
sns.boxplot(ax = axes[1], data = comparison_filt, y = "APhw", x = "recording_protocol")
sns.swarmplot(ax = axes[1], data = comparison_filt, y = "APhw", x = "recording_protocol", color = "black")
sns.boxplot(ax = axes[2], data = comparison_filt, y = "APfallT", x = "recording_protocol")
sns.swarmplot(ax = axes[2], data = comparison_filt, y = "APfallT", x = "recording_protocol", color = "black")
sns.boxplot(ax = axes[3], data = comparison_filt, y = "APriseT", x = "recording_protocol")
sns.swarmplot(ax = axes[3], data = comparison_filt, y = "APriseT", x = "recording_protocol", color = "black")
sns.boxplot(ax = axes[4], data = comparison_filt, y = "slope", x = "recording_protocol")
sns.swarmplot(ax = axes[4], data = comparison_filt, y = "slope", x = "recording_protocol", color = "black")
sns.boxplot(ax = axes[5], data = comparison_filt, y = "APmaxrise", x = "recording_protocol")
sns.swarmplot(ax = axes[5], data = comparison_filt, y = "APmaxrise", x = "recording_protocol", color = "black")
sns.boxplot(ax = axes[6], data = comparison_filt, y = "thr_vol", x = "recording_protocol")
sns.swarmplot(ax = axes[6], data = comparison_filt, y = "thr_vol", x = "recording_protocol", color = "black")

plt.savefig("/mnt/live1/kmikulik/plots/cellchar_spulse_comparison.png")






sns.histplot(data = comparison_filt, x = "APamp",hue = "recording_protocol")
plt.show()
sns.histplot(data = comparison_filt, x = "APhw",hue = "recording_protocol")
plt.show()

sns.histplot(data = comparison_filt, x = "APfallT",hue = "recording_protocol")
plt.show()

sns.histplot(data = comparison_filt, x = "APriseT",hue = "recording_protocol")
plt.show()

sns.histplot(data = comparison_filt, x = "slope",hue = "recording_protocol")
plt.show()

sns.histplot(data = comparison_filt, x = "APmaxrise",hue = "recording_protocol")
plt.show()

sns.histplot(data = comparison_filt, x = "thr_vol",hue = "recording_protocol")
plt.show()







"""
ttest -> can be used, since there are only two means to compare!
"""

APamp_test = pg.ttest(comparison["APamp_x"], comparison["APamp_y"])
thr_vol_test = pg.ttest(comparison["thr_vol_x"], comparison["thr_vol_y"])
slope_test = pg.ttest(comparison["slope_x"], comparison["slope_y"])
APhw_test = pg.ttest(comparison["APhw_x"], comparison["APhw_y"])
APriseT_test = pg.ttest(comparison["APriseT_x"], comparison["APriseT_y"])
APfallT_test = pg.ttest(comparison["APfallT_x"], comparison["APfallT_y"])
APmaxrise_test = pg.ttest(comparison["APmaxrise_x"], comparison["APmaxrise_y"])