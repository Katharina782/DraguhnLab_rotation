#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 30 20:33:20 2021

@author: kmikulik
"""

file = read_file(efile1)

timeline = create_timeline_array()

frame20_vol = file[1][25]

frame20_cur = file[0][25]

fig01, ax1 = plt.subplots(2)
fig01.suptitle("")
ax1[0].plot(timeline, frame20_vol)
ax1[0].set_ylim([-120,50])
ax1[0].set_xlim([390,410])
ax1[0].set_xlabel("time [ms]")
ax1[0].set_ylabel("voltage [mV]")
ax1[1].plot(timeline, frame20_cur)
ax1[1].set_ylim([-200,500])
ax1[1].set_xlim([390,410])
ax1[1].set_ylabel("current [pA]")

time_180 = start_period(file, 25)
degree_list = []
line = ( max(cur) - min(cur) )/2
plt.plot(timeline, frame20_cur)
plt.xlim([0,30])
plt.axhline(y = line)


for i in range(63):
    cur = file[0]
    cur = cur[i]
    curN1 = normalizeSIN(cur)
    curN1 = savgol_filter(curN1, 51, 3)
    line_norm = ( max(curN1) - abs(min(curN1)))/2
    plt.plot(timeline, curN1)
    plt.xlim([0,30])
    plt.axhline(y = line_norm)
    # plus = np.sign(curN1[0:100000])
    # plt.plot(timeline, plus)
    # plt.xlim([0,30])
    plt.show()


line_norm = ( max(curN1) - abs(min(curN1)))/2
plt.plot(timeline, curN1)
plt.xlim([0,30])
plt.axhline(y = line_norm)

min(cur[1 : (15/0.02)])

time = (np.where(timeline < 15))/0.02)


cur()
np.where(cur[time] == min(cur[time]))


ind_180 = np.where (np.diff ( np.sign( curN1[0:100000]) )  == -2  )[0][0]


min(cur)


"""
plot Schritt für Schritt machen
"""