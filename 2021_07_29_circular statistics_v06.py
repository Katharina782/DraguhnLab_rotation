#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 15:49:14 2021

@author: kmikulik
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pingouin as pg
from pingouin import convert_angles
from pingouin import circ_rayleigh
import scipy.stats
from scipy.stats import circstd
import math
#circular mean with scipy
#mean resultant vector lentgh with pingouin
from scipy.stats import circmean

# when running the script on my own laptopo use:
dir_data = "C:\\Users\\Kathi\\Documents\\Heidelberg\\Epyhs_Labrotation_ChristianThome\\PyCharm Analysis\\Analysis\\"

output_dir = "C:\\Users\\Kathi\\Documents\\Heidelberg\\Epyhs_Labrotation_ChristianThome\\PyCharm Analysis\\plots\\"

# when running script on server this is the path to use:
#output_dir = "/mnt/live1/kmikulik/plots/"
#dir_data = "/mnt/live1/kmikulik/Analysis/"


master_map = pd.read_excel (dir_data + "master_map.xlsx")
master_small = pd.read_excel(dir_data + "master_map_small.xlsx")


# """
# we cannot simply calculate the median phase per cell with the common median function.
# Rather we have to use circular statistics.
# """
    
        
# #add row with phases in degree + row with phases in radians   
# for i, row in master_map.iterrows():
#     degree = row["phase"]
#     rad = convert_angles(degree, low = 0, high = 360, positive = True)
#     master_map.at[i, "radians"] = rad

# #name the column that contains phases in degree "degree"        
# master_map.rename(columns = {"phase" : "degree"}, inplace = True)



# """ 
# the mean is the average direction of a variable in the population
# mean resultant vector length is an indication of the spread of the data
# """


# for i, row in master_map.iterrows():
#     cell = row["cell"]
#     freq = row["frequency"]
#     #for freq in [5, 20, 50, 200, 500, 1000]:
#     phase_cell = master_map[ (master_map["cell"] == cell) & (master_map["frequency"] == freq)]["radians"]
#     mean = round(circmean(phase_cell), 4)
#     resultant_vl = round(pg.circ_r(phase_cell), 4)
#     #print(resultant_vl)
#     master_map.at[i, "mean"] = mean
#     master_map.at[i, "resultant_vl"] = resultant_vl



# """
# create a dataframe that contains the mean vector direction and length per cell 
# """

# master_small = master_map.groupby(["cell", "frequency", "mean", "resultant_vl", "AcDnes"]).mean()
# #after groupby the columns are now called index
# #reset columns
# master_small.reset_index(inplace = True)

# master_map.sort_values(by = ["frequency", "cell"], inplace = True)
# master_small.sort_values(by = ["frequency", "cell"], inplace = True)


# def mean_per_cell ():
#     mean_cell = {}
#     for freq in [5, 20, 50, 200, 500, 1000]:
#         for AcDnes in [0,1]:
#             cell_phases = master_map[ (master_map["frequency"] == freq) & (master_map ["AcDnes"] == i) ]["radians"]
#             cell_phases = cell_phases.to_numpy()

#             mean = round(circmean(cell_phases), 4)
#             print(mean)
#             #get the resultant mean vector length
#             resultant_vl = round(pg.circ_r(cell_phases), 4)
#             mean_cell[f"{freq}_{AcDnes}"] = [resultant_vl, mean]
#     return mean_cell
    


    
    
"""
#plot the phases for each frequency and for AcD vs nonAcD separately for all phases of all cells
#the number of bins is determined as the square root of the  number of data points (APs)
The dictionary mean_cell contains frequencies and AcDnes as keys
The values of the dictionary are lists which contain the vector length as a first argument and the mean direction as a second argument
"""

def polar_plot_phase_precision():
    mean_cell = {}
    for freq in [5, 20, 50, 200, 500, 1000]:
        for i in [0, 1]:
            cell_phases = master_map[ (master_map["frequency"] == freq) & (master_map ["AcDnes"] == i) ]["radians"]
            cell_phases = cell_phases.to_numpy()
            #calculate the number of bins for the histogram
            bin_number = round (math.sqrt(len(cell_phases)) )
            #create a histogram
            #hist = number of counts per bin
            #bin-edges = boundaries for each bin
            hist, bin_edges = np.histogram(cell_phases, bins = bin_number)
            bin_edges = (bin_edges[0 : -1])
            binsize = bin_edges[1]-bin_edges[0]
            #by plotting the sqrt of counts you get a rose diagram where the area represents the count
            hist = np.sqrt(hist)
            
            #set figure parameters
            plt.rcParams["figure.figsize"] = [7, 3.5]
            plt.rcParams["figure.autolayout"] = True
            ax = plt.subplot(111, projection = "polar")
            #setting width of bars to binsize -> indicates phase interval of the bin
            bars = ax.bar(bin_edges, hist, width = binsize, edgecolor = "k")
            
            #get the circular mean of this frequency
            mean = round(circmean(cell_phases), 4)
            #get the resultant mean vector length
            resultant_vl = round(pg.circ_r(cell_phases), 4)
            #add the calculated mean direction and vector length to the dictionary
            mean_cell[f"{freq}_{i}"] = [resultant_vl, mean]
            #plot the mean vector with calculated length
            #since the radius of the circle is not one,
            #but equals the maximum value of the counts (hist),
            #we multiply the length * max(hist)
            #ax.plot(mean, resultant_vl * max(hist), color='red', marker='o', markersize=5)
            ax.quiver(0, 0, mean, resultant_vl * max(hist), color='red', angles="xy", scale_units='xy', scale=1., width = 0.015)
    
            for r, bar in zip(hist, bars):
                bar.set_alpha(0.5)
            if i == 0:
                AcDnes = "nonAcD"
            else:
                AcDnes = "AcD"
            number_datapoints = len(cell_phases)
            plt.title(f"{freq} Hertz,{AcDnes}, n = {number_datapoints}")
            
            #save the figures as png format
            png_name = f"{output_dir}{freq} Hertz,{AcDnes}, n = {number_datapoints}.png"
            plt.savefig(png_name ) 
            plt.show()
    return mean_cell
            #return mean_cell    
            
#mean of cell contains a list as value
#first index of the list = resultant_vl
#second index of the list = mean
mean_cell = polar_plot_phase_precision()








"""
the dataframe master_small contains a column mean with the circular mean phase for each cell and each frequency (calculated from all fired action potentials)
A two-way mixed ANOVA is used, because the subjects are different (no rpeated measures).
The mean phase per cell in radians is the dependent variable.
AcDnes and frequency are the between-subject factors

"""




#Now we compare the mean values of all cells between AcD and nonAcD
#two way ANOVA with unbalanced design (unequal sample size)
anova_2 = master_small.anova(dv = "mean", between = ["AcDnes", "frequency"], effsize = "n2").round(3)
anova_2.to_excel(dir_data + "two_way_anova_frequencies.xlsx")

# for the posthoc test to compare the frequencies between AcDnes use a for loop
posthoc_tukey = pd.DataFrame()
for freq in [5, 20, 50, 200, 500, 1000]:
    freq_df = master_map[(master_map["frequency"] == freq)]# & (master_map["AcDnes"] == 1)]
    #nonAcD = master_map[ (master_map["frequency"] == freq) & (master_map["AcDnes"] == 0)]
    #print (AcD)
    posthoc = freq_df.pairwise_tukey(dv= "mean", between = "AcDnes", effsize = "eta-square")
    posthoc["freq"] = freq
    posthoc_tukey = posthoc_tukey.append(posthoc)
    
posthoc_tukey.to_excel("posthoc_tukey_frequencies.xlsx")


#now we compare the mean resultant vector length of all cells between AcD and nonAcD 
anova_2_length = master_small.anova(dv = "resultant_vl", between = ["AcDnes", "frequency"], effsize = "n2").round(3)
anova_2_length.to_excel(dir_data + "two_way_anova_vector_length.xlsx")

# for the posthoc test to compare the resultant vector length between frequencies and between AcDnes use a for loop
posthoc_tukey = pd.DataFrame()
for freq in [5, 20, 50, 200, 500, 1000]:
    freq_df = master_small[ (master_small["frequency"] == freq)]# & (master_map["AcDnes"] == 1)]
    #nonAcD = master_map[ (master_map["frequency"] == freq) & (master_map["AcDnes"] == 0)]
    #print (AcD)
    posthoc = freq_df.pairwise_tukey(dv= "resultant_vl", between = "AcDnes", effsize = "eta-square")
    posthoc["freq"] = freq
    posthoc_tukey = posthoc_tukey.append(posthoc)
    
posthoc_tukey.to_excel(dir_data + "posthoc_tukey_vectorlength.xlsx")
        
  


#boxplot for the anova

#mean vector
sns.boxplot(data = master_small, hue = "AcDnes", y = "mean", x = "frequency").set_ylabel("mean vector direction [rad]")
plt.savefig(output_dir + "boxplot_mean_vector.png")
plt.show()


sns.swarmplot(data = master_small, hue = "AcDnes", y = "mean", x = "frequency").set_ylabel("mean vector direction [rad]")
plt.savefig(output_dir + "swarmplot_mean_vector.png")
plt.show()


#resultant vector length
sns.boxplot(data = master_small, hue = "AcDnes", y = "resultant_vl", x = "frequency").set_ylabel ("resultant vector length")
plt.savefig(output_dir + "boxplot_resultant_vl.png")
plt.show()

sns.swarmplot(data = master_small, hue = "AcDnes", y = "resultant_vl", x = "frequency").set_ylabel ("resultant vector length")
plt.savefig(output_dir + "swarmplot_resultant_vl.png")
plt.show()

    

"""
partial regression
"""
partial_regression = pd.DataFrame()
for freq in [5, 20, 50, 200, 500, 1000]:
    freq_df = master_small[ (master_small["frequency"] == freq)]
    part_reg = pg. partial_corr(data = freq_df, x = "AcDnes", y = "resultant_vl", method = "spearman").round(3)
    part_reg ["frequency"] = freq
    partial_regression = partial_regression.append(part_reg)
    
partial_regression.to_excel(dir_data + "partial_regressioin_vector_direction_lenght.xlsx")

partial_regression = pd.DataFrame()
for freq in [5, 20, 50, 200, 500, 1000]:
    freq_df = master_small[ (master_small["frequency"] == freq)]
    part_reg = pg. pairwise_corr(data = freq_df, columns = ["AcDnes", "resultant_vl"], method = "spearman").round(3)
    part_reg ["frequency"] = freq
    partial_regression = partial_regression.append(part_reg)
    



partial_regression = pd.DataFrame()
for freq in [5, 20, 50, 200, 500, 1000]:
    freq_df = master_small[ (master_small["frequency"] == freq)]
    part_reg = pg. partial_corr(data = freq_df, x = "mean", y = "AcDnes_continuum", covar = ["AcDnes"], method = "spearman").round(3)
    part_reg ["frequency"] = freq
    partial_regression = partial_regression.append(part_reg)












"""
calculate the grand mean & plot it
"""


        
"""
claculate the mean of means with formula from Biostatistical analysis book
the mean vector has rectangular coordinates that are calculated below. You cannot simply compute the mean with the above formula, 
because this formula assumes that all values have a vector length = 1.
However, this is not the case for the mean vectors we calculated -> they have values different from 1, indicating the spread 
of the original data around the mean vector
"""


#the formula to get the rectangular coordinates of the grand mean is:
# X = sum(r * cos(a)) / k
def get_x_coordinate(cos_mean, sin_mean, resultant_vl):
    product_list = []
    for i in range(0, len(cos_mean)):
        cos = cos_mean[i]
        length = resultant_vl[i]
        product = cos * length
        product_list.append(product)
        
    x_coord = sum(product_list) / len(product_list)
    return x_coord

def get_y_coordinate(cos_mean, sin_mean, resultant_vl):
    product_list = []
    for i in range(0, len(cos_mean)):
        sin = sin_mean[i]
        length = resultant_vl[i]
        product = sin * length
        product_list.append(product)
    
    y_coord = sum(product_list) / len(product_list)
    return y_coord

"""
this function computes the mean of mean vectors using the above functions to get the rectangular coordinates
and the direction of the grand mean vector
It returns a dictionary "mean_vectors" that contains  as value a list with the length of the vectors as first argument of the list and 
the angle (direction) of the vector as a second argument
The keys of the dictionary are the respective frequencies and AcDnes
"""
def mean_of_means():
    mean_vectors = {}
    for freq in [5, 20, 50, 200, 500, 1000]:
        for AcDnes in [0,1]:
            # get the mean direction
            
            mean_per_cell = master_small[ (master_small["frequency"] == freq) & (master_small ["AcDnes"] == AcDnes) ]["mean"]
            #get the resultant lenght of the vecotr
            resultant_vl = master_small[ (master_small["frequency"] == freq) & (master_small ["AcDnes"] == AcDnes) ]["resultant_vl"].tolist()
            cos_mean = np.cos(mean_per_cell).tolist()
            sin_mean = np.sin(mean_per_cell).tolist()
            #get rectangular coordinates of the grand mean
            x_coord = get_x_coordinate(cos_mean, sin_mean, resultant_vl)
            y_coord = get_y_coordinate(cos_mean, sin_mean, resultant_vl)
            #length of mean resultant vector
            len_r = math.sqrt((x_coord**2 + y_coord**2))
            #for different quadrants of the circle you need to use different inverse cos/sin/tan functions
            if freq < 500:
                #angle/direction of the mean resultant vector
                angle = np.arccos ( (x_coord / len_r) )# + math.pi
                #now values is located in the third quadrant so we have to add 180°
                angle = np.arctan(y_coord/x_coord) + 3.141592653589793
            if freq == 1000:
                angle = np.arcsin(y_coord/len_r) #+ math.pi
            else:
                 #now values is located in the third quadrant so we have to add 180°
                angle = np.arctan(y_coord/x_coord) + 3.141592653589793
                
            #angle1 = np.arcsin( (y_coord/ len_r) ) 
            mean_vectors [f"{freq}_{AcDnes}"] = [len_r, angle]#, angle1]
    return mean_vectors

mean_vectors = mean_of_means()



"""
plot the mean phase of all cells of one frequency and either AcD or nonAcD
"""
def polar_plot_grand_mean():
    for freq in [5, 20, 50, 200, 500, 1000]:
        for i in [0, 1]:
            cell = master_small[ (master_small["frequency"] == freq) & (master_small ["AcDnes"] == i) ]["mean"]
            cell = cell.to_numpy()
            #calculate the number of bins for the histogram
            bin_number = round (math.sqrt(len(cell)) )
            #create a histogram
            #hist = number of coutns for each bin
            #bin_edges = boundaries of each bin
            hist, bin_edges = np.histogram(cell, bins = bin_number)
            bin_edges = (bin_edges[0 : -1])
            binsize = bin_edges[1]-bin_edges[0]
            #by plotting the sqrt of counts you get a rose diagram where the area represents the count
            hist = np.sqrt(hist)
            
            #set figure parameters
            plt.rcParams["figure.figsize"] = [7, 3.5]
            plt.rcParams["figure.autolayout"] = True
            ax = plt.subplot(111, projection = "polar")
            #setting width of bars to binsize -> indicates phase interval of the bin
            bars = ax.bar(bin_edges, hist, width = binsize, edgecolor = "k")
            
            #calculate the grand mean vector -> function computes a dictionary with grand mean per frequency & AcDnes
            mean_vectors = mean_of_means()
            #get the grand mean of this frequency from the mean_vectors dictionary
            mean = round( mean_vectors[f"{freq}_{i}"][1] , 4)
            #print(mean)
            #get the grand mean vector length for this frequency
            resultant_vl = round(mean_vectors[f"{freq}_{i}"][0] , 4)
            #plot the mean vector with calculated length
            #sinze the radius of the circle is not one,
            #but equals the maximum value of the counts (hist),
            #we multiply the length * max(hist)
            #ax.plot(mean, resultant_vl * max(hist), color='red', marker='o', markersize=5)
            ax.quiver(0, 0, mean, resultant_vl * max(hist), color='red', angles="xy", scale_units='xy', scale=1., width = 0.015)
            
            for r, bar in zip(hist, bars):
                #the bar colour is set to transparent yelllow
                bar.set_facecolor("yellow")
                bar.set_alpha(0.5)
            if i == 0:
                AcDnes = "nonAcD"
            else:
                AcDnes = "AcD"
            number_datapoints = len(cell)
            
            plt.title(f"{freq} Hertz, {AcDnes}, n = {number_datapoints}")
            #save the plots as png format
            png_name = f"{output_dir}mean phase of every cell,{freq} Hertz,{AcDnes}, n = {number_datapoints}.png"
            plt.savefig(png_name ) 
            plt.show()
            
            
polar_plot_grand_mean()
            





















    



        
        
        
        
        
        
        
        
        
        
        
        
        
   