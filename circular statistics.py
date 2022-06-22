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


master_map = pd.read_excel ("/mnt/live1/kmikulik/Analysis/master_map.xlsx")
master_small = pd.read_excel("/mnt/live1/kmikulik/Analysis/master_map_small.xlsx")


#funciton to get only phase values for a certain frequency for either Acd or nonAcD cells
def get_phase (freq):
    nonAcD = master_small[ (master_small["frequency"] == freq) & (master_small["AcDnes"] == 0)]["median_phase"]
    AcD = master_small[ (master_small["frequency"] == freq) & (master_small["AcDnes"] == 1)]["median_phase"]
    nonAcD = nonAcD.to_numpy()
    AcD = AcD.to_numpy()
    return nonAcD, AcD



#define dataframes
non_5, acd_5 = get_phase(5)
non_20, acd_20 = get_phase(20)
non_50, acd_50 = get_phase(50)
non_200, acd_200 = get_phase(200)
non_500, acd_500 = get_phase(500)
non_1000, acd_1000 = get_phase(1000) 

sep_phase = [non_1000,non_20, non_200, non_5, non_50, non_500, acd_1000, acd_20, acd_200, acd_5, acd_50, acd_500]

def convert_numpy(y):
    x = y.to_numpy()
    return x

non_5 = non_5.to_numpy()
acd_5 = acd_5.to_numpy()
non_20 = convert_numpy(non_20)
acd_20 = convert_numpy(acd_20)
non_50 = convert_numpy(non_50)
acd_50 = convert_numpy(acd_50)
acd_200 = convert_numpy(acd_200)
non_200 = convert_numpy(non_200)
acd_500 = convert_numpy(acd_500)
non_500 = convert_numpy(non_500)
acd_1000 = convert_numpy(acd_1000)
non_1000 = convert_numpy(non_1000)

phase_df = pd.DataFrame(columns=["5", "20", "50", "200", "500", "1000"])
for i in [5, 20, 50, 200, 500, 1000]:
    degrees = master_map[master_map["frequency"] == i ]["phase"]
    phase_df[f"{i}"] = degrees


small_df = master_small[master_small["frequency"] == 5] ["median_phase"]
small_array = small_df.to_numpy()

phase_array = phase_df["5"].to_numpy()

#convert degrees to radians
radians = convert_angles(phase_array, low = 0, high = 360)
radians_small = convert_angles(small_array, low = 0, high = 360)
mean_radians = round(pg.circ_mean(radians), 4)

""" calculate the mean vector length
The mean resultant vector is a quantity for measurement of circular spread or hypothesis testing in directional statistics.
The closer it is to one, the more concentrated the sample is around the mean direction.
"""
mean_vector_length = pg.circ_r(radians)

"""circular standard deviation"""
std = round(np.sqrt(-2*np.log(mean_vector_length)), 4)
#built in function of scipy
std_scipy = round(circstd(radians), 4)


""" Rayleigh test for non-uniformity of circular data"""
z, pval = circ_rayleigh(radians)
print(round(z, 3), round(pval, 6))

z, pval = circ_rayleigh(radians_small)

print(round(z, 3), round(pval, 6))