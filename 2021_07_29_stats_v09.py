#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 10:14:13 2021

@author: kmikulik
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pingouin as pg

dir_data = "C:\\Users\\Kathi\\Documents\\Heidelberg\\Epyhs_Labrotation_ChristianThome\\PyCharm Analysis\\Analysis\\"
output_dir = "C:\\Users\\Kathi\\Documents\\Heidelberg\\Epyhs_Labrotation_ChristianThome\\PyCharm Analysis\\plots\\"

# when running script on server this is the path to use:
#output_dir = "/mnt/live1/kmikulik/plots/"
#dir_data = "/mnt/live1/kmikulik/Analysis/"

master_map = pd.read_excel(dir_data + "master_map.xlsx")
master_small = pd.read_excel(dir_data + "master_map_small.xlsx")
"""
for any analysis that does not take into account the phase precision we can make the dataframe even smaller
In master_cell there is only one row for each cell
"""

master_cell = pd.read_excel(dir_data + "master_cell.xlsx")


"""
this function generates a separate histogram for each frequency and each AcDnes
"""
# #you should add a range for xlim and ylim here in order to be able to better compare the two gaphs in parallel
# def histplots_freq ():
#     for freq in [5, 20, 50, 200, 500, 1000]:
#         cellAcD = master_map[(master_map["frequency"] == freq) & (master_map["AcDnes"] == 1)]
#         nonAcD = master_map[(master_map["frequency"] == freq) & (master_map["AcDnes"] == 0)]
#         nbins_acd = int( np.sqrt(len(cellAcD)) )
#         nbins_non = int( np.sqrt(len(nonAcD)) )
#         sns.displot(data=cellAcD, x="degree", bins=nbins_acd).set(title=f"{freq}_AcD", xlim = (0,360))
#         plt.show()
#         sns.displot(data=nonAcD, x="degree", bins=nbins_non).set(title=f"{freq}_nonAcD", xlim = (0,360))
#         plt.show()
# histplots_freq()




# the number of bins for each histogram was determined as the square root of the number of samples
# the following function returns a dictionary with a dataframe of all Aps of one frequencies
# and the number of bins calculated for each frequency to be used for histograms later on
def freq_bins_dict():
    data = {}
    for freq in [5, 20, 50, 200, 500, 1000]:
        cell = master_map[(master_map["frequency"] == freq)]
        nbins = int(np.sqrt(len(cell)))
        data[freq] = [cell, nbins]
    return data


df_freq = freq_bins_dict()




"""
the following code creates acombined plot with histograms for all frequencies
"""
# plot a 2x3 plot of the distribution of the phase of the sin curve on which APs are fired
# separate plots for each frequency
# AcDnes is binary: AcD-cell = 1, non-AcD cell = 0
fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharey=False)
fig.suptitle("phase precision of all fired action potentials for all cells for different frequencies (non-AcDcell = 0, AcD cell = 1)")
sns.histplot(ax=axes[0, 0], data=df_freq[5][0], x="degree",
             bins=df_freq[5][1], hue="AcDnes").set(title="5 Hz", xlim=(0, 360))
sns.histplot(ax=axes[0, 1], data=df_freq[20][0], x="degree",
             bins=df_freq[20][1], hue="AcDnes").set(title="20 Hz", xlim=(0, 360))
sns.histplot(ax=axes[0, 2], data=df_freq[50][0], x="degree",
             bins=df_freq[50][1], hue="AcDnes").set(title="50 Hz", xlim=(0, 360))
sns.histplot(ax=axes[1, 0], data=df_freq[200][0], x="degree",
             bins=df_freq[200][1], hue="AcDnes").set(title="200 Hz", xlim=(0, 360))
sns.histplot(ax=axes[1, 1], data=df_freq[500][0], x="degree",
             bins=df_freq[500][1], hue="AcDnes").set(title="500 Hz", xlim=(0, 360))
sns.histplot(ax=axes[1, 2], data=df_freq[1000][0], x="degree",
             bins=50, hue="AcDnes").set(title="1000 Hz", xlim=(0, 360))
plt.savefig(output_dir + "phase_precision_histograms.png")


"""
boxplot with frequencies for anova and posthoc tukey
"""
sns.boxplot(data = master_map, x = "frequency", y = "degree", hue = "AcDnes")
plt.savefig(output_dir + "boxplot_degree_frequency.png")





"""
partial correlation with AP parameters and histo data
"""

#spearman partial correlation
corr_df = master_cell.drop(columns = ["cell", 'Unnamed: 0', 'apical_dendrite1', 'apical_dendrite2',
       'apical_dendrite3', 'apical_dendrite4', 'apical_dendrite5', "AcDnes_continuum", "time_since_zero"])

#create 3 separate correlation tables

#first table contains correlations within morphology
morph = corr_df.drop(columns = ["rmp", "input_res", "Ih_vol", "thr_vol", "rheo","burstratio", "slope", "APamp",  'APhw', 'APriseT',
       'APfallT', 'APmaxrise', 'age(days)', 'Virus injection', 'end_acc_res',
       'AcDnes'])

corr_morph = pg.rcorr(morph, method = "spearman", upper = "pval", decimals = 3)
corr_morph.to_excel(dir_data + "corr_morph_spearman.xlsx")



#second table contains correlations within electrophysiology
ephys = corr_df.drop(columns = ['AIS_length', 'AcD_stem_length', 'AcD_stem_width',
       'ais_distance_from_last_branch', 'ais_distance_from_soma',
       'mean_apical_dendrite', 'soma_area','age(days)', 'Virus injection', 'end_acc_res',
       'AcDnes'])

corr_ephys = pg.rcorr(ephys, method = "spearman", upper = "pval", decimals = 3)
corr_ephys.to_excel(dir_data + "corr_ephys_spearman.xlsx")




# corr_matrix = pg.rcorr(corr_df, method = "spearman", upper = "pval", decimals = 3 )

# corr_matrix.to_excel("/mnt/live1/kmikulik/Analysis/corr_matrix_spearman.xlsx")


# #pearson partial correlation
# corr_matrix_pearson = pg.rcorr(corr_df, method = "pearson", upper = "pval", decimals = 3 )

# corr_matrix_pearson.to_excel("/mnt/live1/kmikulik/Analysis/corr_matrix_pearson.xlsx")


# #Spearman pairwise correlatioin
# corr = pg.pairwise_corr(corr_df, method='spearman', padjust='bonf').round(4)

# table = pd.DataFrame(corr)
# table = table[['X', 'Y', 'n', 'z', 'p-unc', 'p-corr']]
# table = table[table['p-corr'] < 0.06]

# table.to_excel("/mnt/live1/kmikulik/Analysis/corr_table_spearman.xlsx")


# # Pearson pairwise correlation
# corr = pg.pairwise_corr(corr_df, method='pearson', padjust='bonf').round(4)

# table_pearson = pd.DataFrame(corr)
# table_pearson = table[['X', 'Y', 'n', 'z', 'p-unc', 'p-corr']]
# table_pearson = table[table['p-corr'] < 0.06]

# table_pearson.to_excel("/mnt/live1/kmikulik/Analysis/corr_table_pearson.xlsx")





"""
the verbist paper showed that with increasing AIS  distance the somatic AP waveform became steeper
here I found the opposite
"""

AcD = master_cell[master_cell["AcDnes"] == 1]
nonAcD = master_cell[master_cell["AcDnes"] == 0]
# slope is significantly different between AcD and nonAcD
slope_test = pg.ttest(AcD["slope"], nonAcD["slope"]).round(3)
slope_test.to_excel(dir_data + "slope_ttest.xlsx")
# amp not significantly different
amp_test = pg.ttest(AcD["APamp"], nonAcD["APamp"]).round(3)
# somasize is significantly different
soma_test = pg.ttest(AcD["soma_area"], nonAcD["soma_area"]).round(3)


sns.boxplot(data=master_cell, x="AcDnes", y="slope")
plt.savefig(output_dir + "slope_AcD.png")


sns.boxplot(data=master_cell, x="AcDnes", y="slope")
















"""
visualization AP parameter - AcDcells vs. nonAcDcells
"""
fig, axes = plt.subplots(3, 4, figsize=(35, 20), sharey=False)
fig.suptitle("action potential parameter (nonAcD cell = 0, AcD cell = 1)").set_size(50)
sns.boxplot(ax = axes[0, 1], data = master_cell, y = "ais_distance_from_soma", x = "AcDnes", )
sns.boxplot(ax = axes[0, 2], data = master_cell, y = "soma_area", x = "AcDnes")
sns.boxplot(ax = axes[0, 3], data = master_cell, y = "APamp", x = "AcDnes")
sns.boxplot(ax = axes[0, 0], data = master_cell, y = "AIS_length", x = "AcDnes")
sns.boxplot(ax = axes[2, 0], data = master_cell, y = "burstratio", x = "AcDnes")
sns.boxplot(ax = axes[2, 1], data = master_cell, y = "rheo", x = "AcDnes")
sns.boxplot(ax = axes[2, 2], data = master_cell, y = "Ih_vol", x = "AcDnes")
sns.boxplot(ax = axes[2, 3], data = master_cell, y = "APhw", x = "AcDnes")
sns.boxplot(ax = axes[1, 0], data = master_cell, y = "thr_vol", x = "AcDnes")
sns.boxplot(ax = axes[1, 1], data = master_cell, y = "slope", x = "AcDnes")
sns.boxplot(ax = axes[1, 2], data = master_cell, y = "APriseT", x = "AcDnes")
sns.boxplot(ax = axes[1, 3], data = master_cell, y = "APfallT", x = "AcDnes")
for i in range(0, 3):
    for y in range(0,4):
        axes[i, y].set_xlabel("AcDnes").set_size(30)
        #x = [0,1]
        #axes[i, y].set_xticklabels(x, fontsize = 25)
        axes[i,y].tick_params(axis='both', which='major', labelsize=25)

axes[0,1].set_ylabel("ais_distance [µm]").set_size(30)
axes[0,2].set_ylabel("soma_area [µm²]").set_size(30)
axes[0,3].set_ylabel("AP amplitude [mV]").set_size(30)
axes[0,0].set_ylabel("AIS_length [µm]").set_size(30)
axes[2,0].set_ylabel("burstratio").set_size(30)
axes[2,1].set_ylabel("rheobase [pA]").set_size(30)
axes[2, 2].set_ylabel("Ih_sag").set_size(30)
axes[2, 3].set_ylabel("AP halfwidth").set_size(30)
axes[1, 0].set_ylabel("AP voltage threshold [mV]").set_size(30)
axes[1, 1].set_ylabel("maximum AP  slope [mV/ms]").set_size(30)
axes[1, 2].set_ylabel("AP rise time").set_size(30)
axes[1, 3].set_ylabel("AP fall time").set_size(30)
plt.show()

plt.savefig(output_dir + "AP_parameter_boxplots.png")





"""
visualization AP parameter - AcDcells vs. nonAcD cells - boxplots + SWARMPLOTS
"""

fig, axes = plt.subplots(3, 4, figsize=(35, 20), sharey=False)
fig.suptitle("action potential parameter (nonAcD cell = 0, AcD cell = 1)").set_size(50)
sns.boxplot(ax = axes[0, 1], data = master_cell, y = "ais_distance_from_soma", x = "AcDnes" )
sns.swarmplot (ax = axes[0, 1], data = master_cell, y = "ais_distance_from_soma", x = "AcDnes", color = "black")

sns.boxplot(ax = axes[0, 2], data = master_cell, y = "soma_area", x = "AcDnes")
sns.swarmplot(ax = axes[0, 2], data = master_cell, y = "soma_area", x = "AcDnes", color = "black")

sns.boxplot(ax = axes[0, 3], data = master_cell, y = "APamp", x = "AcDnes")
sns.swarmplot(ax = axes[0, 3], data = master_cell, y = "APamp", x = "AcDnes", color = "black")

sns.boxplot(ax = axes[0, 0], data = master_cell, y = "AIS_length", x = "AcDnes")
sns.swarmplot(ax = axes[0, 0], data = master_cell, y = "AIS_length", x = "AcDnes", color = "black")

sns.boxplot(ax = axes[2, 0], data = master_cell, y = "burstratio", x = "AcDnes")
sns.swarmplot(ax = axes[2, 0], data = master_cell, y = "burstratio", x = "AcDnes", color = "black")

sns.boxplot(ax = axes[2, 1], data = master_cell, y = "rheo", x = "AcDnes")
sns.swarmplot(ax = axes[2, 1], data = master_cell, y = "rheo", x = "AcDnes", color = "black")

sns.boxplot(ax = axes[2, 2], data = master_cell, y = "Ih_vol", x = "AcDnes")
sns.swarmplot(ax = axes[2, 2], data = master_cell, y = "Ih_vol", x = "AcDnes", color = "black")

sns.boxplot(ax = axes[2, 3], data = master_cell, y = "APhw", x = "AcDnes")
sns.swarmplot(ax = axes[2, 3], data = master_cell, y = "APhw", x = "AcDnes", color = "black")

sns.boxplot(ax = axes[1, 0], data = master_cell, y = "thr_vol", x = "AcDnes")
sns.swarmplot(ax = axes[1, 0], data = master_cell, y = "thr_vol", x = "AcDnes", color = "black")

sns.boxplot(ax = axes[1, 1], data = master_cell, y = "slope", x = "AcDnes")
sns.swarmplot(ax = axes[1, 1], data = master_cell, y = "slope", x = "AcDnes", color = "black")

sns.boxplot(ax = axes[1, 2], data = master_cell, y = "APriseT", x = "AcDnes")
sns.swarmplot(ax = axes[1, 2], data = master_cell, y = "APriseT", x = "AcDnes", color = "black")

sns.boxplot(ax = axes[1, 3], data = master_cell, y = "APfallT", x = "AcDnes")
sns.swarmplot(ax = axes[1, 3], data = master_cell, y = "APfallT", x = "AcDnes", color = "black")

for i in range(0, 3):
    for y in range(0,4):
        axes[i, y].set_xlabel("AcDnes").set_size(30)
        #x = [0,1]
        #axes[i, y].set_xticklabels(x, fontsize = 25)
        axes[i,y].tick_params(axis='both', which='major', labelsize=25)

axes[0,1].set_ylabel("ais_distance [µm]").set_size(30)
axes[0,2].set_ylabel("soma_area [µm²]").set_size(30)
axes[0,3].set_ylabel("AP amplitude [mV]").set_size(30)
axes[0,0].set_ylabel("AIS_length [µm]").set_size(30)
axes[2,0].set_ylabel("burstratio").set_size(30)
axes[2,1].set_ylabel("rheobase [pA]").set_size(30)
axes[2, 2].set_ylabel("Ih_sag").set_size(30)
axes[2, 3].set_ylabel("AP halfwidth").set_size(30)
axes[1, 0].set_ylabel("AP voltage threshold [mV]").set_size(30)
axes[1, 1].set_ylabel("maximum AP  slope [mV/ms]").set_size(30)
axes[1, 2].set_ylabel("AP rise time").set_size(30)
axes[1, 3].set_ylabel("AP fall time").set_size(30)

plt.savefig(output_dir + "AP_parameter_boxplots_swarmplots.png")










"""
ttest -> can be used, since there are only two means to compare!
create two small dataframes, one for AcD cells, one for nonAcD cells
"""
AcDcells = master_cell[master_cell["AcDnes"] == 1]
nonAcDcells = master_cell[master_cell["AcDnes"] == 0]


ais_distance_test = pg.ttest(AcDcells["ais_distance_from_soma"] , nonAcDcells[ "ais_distance_from_soma"]).round(3)
ais_distance_test["parameter"] = "ais_distance_from_soma"
soma_area_test = pg.ttest(AcDcells["soma_area"] , nonAcDcells[ "soma_area"]).round(3)
soma_area_test["parameter"] = "soma_area"
APamp_test = pg.ttest(AcDcells["APamp"] , nonAcDcells[ "APamp"]).round(3)
APamp_test["parameter"] = "APamp"
ais_length_test = pg.ttest(AcDcells["AIS_length"] , nonAcDcells[ "AIS_length"]).round(3)
ais_length_test["parameter"] = "AIS_length"
rheo_test = pg.ttest(AcDcells["rheo"] , nonAcDcells[ "rheo"]).round(3)
rheo_test["parameter"] = "rheobase"
Ih_vol_test = pg.ttest(AcDcells["Ih_vol"] , nonAcDcells[ "Ih_vol"]).round(3)
Ih_vol_test["parameter"] = "Ih_sag"
APhw_test = pg.ttest(AcDcells["APhw"] , nonAcDcells[ "APhw"]).round(3)
APhw_test["parameter"] = "APhw"
thr_vol_test = pg.ttest(AcDcells["thr_vol"] , nonAcDcells[ "thr_vol"]).round(3)
thr_vol_test["parameter"] = "thr_vol"
slope_test = pg.ttest(AcDcells["slope"] , nonAcDcells[ "slope"]).round(3)
slope_test["parameter"]  = "slope"
APriseT_test = pg.ttest(AcDcells["APriseT"] , nonAcDcells[ "APriseT"]).round(3)
APriseT_test["parameter"] = "APriseT"
APfallT_test = pg.ttest(AcDcells["APfallT"] , nonAcDcells[ "APfallT"]).round(3)
APfallT_test["parameter"] = "APfallT"

AP_parameter_ttests = pd.DataFrame()
AP_parameter_ttests = AP_parameter_ttests.append([ais_distance_test, soma_area_test, APamp_test, ais_length_test, rheo_test, Ih_vol_test, 
                            APhw_test, thr_vol_test, slope_test, APriseT_test, APfallT_test] )
AP_parameter_ttests.to_excel(dir_data + "AP_parameter_ttests.xlsx")









# """
# correlation plots
# """
# sns.lmplot(x="soma_area", y="AIS_length", hue="AcDnes", data=master_cell)


# master_tiny = master_small.groupby(["cell"], as_index=False).median()

# """ morphology """
# # AIS length increases more rapidly with somasize for AcD cells
# sns.lmplot(x="somasize", y="AIS_length", hue="AcDnes", data=master_tiny)

# # decreasing somasize -> more distant AIS for AcD
# # increasing distance with increasing somasize for nonAcD
# sns.lmplot(x="somasize", y="AIS_distance_from_Soma",
#            hue="AcDnes", data=master_tiny)
# sns.lmplot(x="somasize", y="APamp",  hue="AcDnes", data=master_tiny)
# sns.lmplot(x="AIS_length", y="AIS_distance_from_Soma",
#            hue="AcDnes", data=master_tiny)
# sns.lmplot(x="AIS_length", y="APamp",  hue="AcDnes", data=master_tiny)

# # positive correlation for AcD
# # negative correlation for non-AcD
# sns.lmplot(x="AIS_distance_from_Soma", y="APamp",
#            hue="AcDnes", data=master_tiny)
# sns.lmplot(x="AIS_length", y="burstratio",  hue="AcDnes", data=master_tiny)
# sns.lmplot(x="somasize", y="burstratio",  hue="AcDnes", data=master_tiny)

# # AcD stem width remains relatively stable while stem length increases
# sns.lmplot(x="AcD_stem_length", y="AcD_stem_width",
#            hue="AcDnes", data=master_tiny)

# # stem length increases with somasize
# sns.lmplot(x="somasize", y="AcD_stem_length", hue="AcDnes", data=master_tiny)

# # this plot has a vew outliers that make the data look strange
# # With increasing stem length the
# sns.lmplot(x="AIS_distance_from_Soma", y="AcD_stem_length",
#            hue="AcDnes", data=master_tiny)






