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

    
efile1 = '/mnt/archive1/cthome/Backup_ProjectArchive/Y_AISdiversity_PYR/Kathi/' +\
    '23_02_2021/MKcellchar_021.cfs'
    
import stfio

from scipy.signal import find_peaks
import math

#from scipy.optimize import curve_fit
import os

"""
x = np.arange(0, 20, 0.5)
y = pow(x, 2)
print(x)
print(y)
plt.plot(x, y)

numbers = [0, 1, 2, 3, "peter"]
print(numbers)
numbers_2 = np.array(numbers)
print(numbers_2)

x = x.reshape(5,8)
print(x)

print(rec[1, ])
plt.plot(timeline, rec[1,])
"""

rec = stfio.read(efile1,"tcfs")
#convert data into an array
rec = np.array(rec)

vol = rec[1]

timeline = np.arange(0,2000,0.02).tolist()
x_time = np.asarray(timeline)
print(x_time)
print(np.where((x_time > 800) & (x_time < 1300))
            


#holding current
#only current recordings
cur = rec[0]
#only datapoints from first frame
cur0 = cur[0]
#print(cur)
#we want to know the average current between 410ms and 790ms = holding current
#convert time to data points
pa1 = int(410/0.02)
pa2 = int(790/0.02)
#average holding current for first frame
add = 0
count = 0
for i in range (pa1, pa2):
    #add all current values
    #cur[i] gives me back the voltage in mV for each datapoint i 
    add += cur0[i]
    count += 1
average = add/count
hold = average
print("Holding curent: ", hold)

#average of holding current for several frames
# add = 0
# count = 0
# add1 = 0
# count1 = 0

# for i in range (0, 32):
#     cur = cur [i]
#     for i in range (pa1, pa2):
#     #print(cur[i])
#     #add all current values
#         add = add + cur[i]
#         count = count + 1
#     add1 = add + add1
#     count1 = count + count1
# average = add/count
# print ("holding current:", average, "pA")


#determine input resistance

#determine delta voltage 

#voltage at first 25 pA step
# -25pA in frame 7
#  25pA in frame 9
v = rec[1]
#only frame 7 of voltage recordings
v = v[7]
#define the range where we determine the average potential 
#between 900ms and 1250 ms
#convert time to datapoints
v1 = int(900/0.02)
v2 = int(1250/0.02)

add = 0
count = 0

for i in range (v1, v2):
    add = add + v[i]
    count = count + 1
v25 = add/count
#print(v25)


#resting membrane potential
#selbes Intervall wie oben bei holding current (pa1, pa2)
count = 0
add = 0
for i in range (pa1, pa2):
    add = add + v[i]
    count = count + 1
rmp = add/count
print("Resting membrane potential: ", rmp)

#delta voltage = difference between resting voltage and voltage at 25pA step
delta = abs(rmp) - abs(v25)
print("delta U: ", delta)
print("delta in pA: ", 25)

#calculate resistance
#R = U/I
#pA -> mA
I = 25/1000000000
#print(I)
#in ohm
inres = delta/I
#print(inres)
#in Mohm
inres = inres/1000000
print ("Input resistance in Mega Ohm: ", inres)







#find peaks returns ndarray -> indices of peaks tkhat satisfy all given conditions
#count AP per frame
pA = -225 
f = list()
c = list()
l = list() 
t = list()
p = list()
#in order to specify a directory where the plots are saved:
frame = "frame"
pdf = ".pdf"
inp_out = "input-output"
fig_dir = "/mnt/live1/kmikulik/plots"

#in order to be able to use the following code for any file
#leave the number of frames open
x = len(vol)
total = 0
for i in range(0, x):
    fig01, ax1 = plt.subplots(2) # sharey = sharing of y axis beschriftung
    fig01.suptitle(i)
    ax1[0].plot(timeline, vol[i])
    ax1[0].set_ylim([-120, 50])
    ax1[0].set_xlabel("time [ms]")
    ax1[0].set_ylabel("voltage [mV]")
    #axs[0].plot(x_time, np.zeros_like(y_voltage),'--',color='gray')
    ax1[1].plot(timeline, cur[i]) 
    ax1[1].set_ylim([-200, 500])
    #ax1[1].set_xlabel("time [ms]")
    ax1[1].set_ylabel("curent [pA]")
    #plt.plot(timeline, vol[i,])
    peaks, _ = find_peaks(vol[i], height=0)
    count = 0
    total = total + count
    #print(total)
    pA = pA + 25
    for peak in peaks:
        #peaks = array of integers
        #p.append creates a list with length = number of total APs
        #contains alls datapoints at which an AP occurs
        p.append(peak)
        #ax1[0].plot(timeline[peak], peak[peak],'x')
        ax1[0].axvline(x=timeline[peak])
        count = count + 1
        #to know the total number of detected APs so far:
        #add one to total for every peak, BUT do not reset to zero after each loop
        total = total +1
    #save the plot in the specified directory and title plot with frame and number
    fig01.savefig(f"{fig_dir}/{frame}{i}{pdf}")
    #f"{kathi}/{date}/{file}"
    #plt.title(i)
    # plt.axis([0, 2000, -120, 50])
    # plt.xlabel("time [ms]")
    # plt.ylabel("potential [mV]")
    # plt.show()a
    
    f.append(i)
    l.append(count)
    c.append(pA)
    t.append(total)
    print(i, count, pA) 
    #print("frame: ", i, "number of APs: ", count, "current injection: ", pA)
    #print(i, count, total, pA)  
    
#print(f, l, c)  


#plot AP count per frame
fig02, ax2 = plt.subplots()
ax2.plot(c, l, "bo", color = "red", markersize = 3)
ax2.set_title("input - output")
ax2.set_xlabel("injected current [pA]")
ax2.set_ylabel("number of action potentials")
fig02.savefig(f"{fig_dir}/{inp_out}{pdf}")
#plt.plot(c, l, "bo", color = "red", markersize = 3)
# plt.title("input - output")
# plt.xlabel("injected current [pA]")
# plt.ylabel("number of action potentials")
    
#Rheobase
#Differenz Haltestrom - Strominjektion 
for i in l:
    if i < 1:
        continue
    if i >= 1:
        #print(i)
        pos = l.index(i)
        #print (pos)
        rheo = c[pos]
        print("Rheobase:", rheo)
        break
        
        #funktioniert für positiven und negativen Haltestrom, weil der Betrag abgezogen wird
        #rheo =(inj - abs(hold))
        
 
#Interspike Intervall
#take the frame that has at least 7 spikes
for i in l:
    if i < 7:
        continue
    if i >= 7:
        #tell me the frame 
        frame = l.index(i)
        print(frame)
        break
#now we know with which frame to work
#to find positions of peaks 1, 2, 6, 7 we have to determine the index of the peaks
#number of total AP is one bigger than index of list p
p1 = t[frame] - l[frame] 
p2 = p1 + 1
p6 = p1 + 5
p7 = p1 + 6
print(p1, p2, p6, p7)
#take peak datapoints
peak1 = p[p1]
peak2 = p[p2]
peak6 = p[p6]
peak7 = p[p7]
print(peak1, peak2, peak6, peak7)
#calculate interspike interval (isi)
isi1 = peak2 - peak1
print(isi1)
isi2 = peak7 -peak6
print(isi2)
timeint1 = isi1 * 0.02
timeint2 = isi2 * 0.02
ISI = timeint2/timeint1
print("ISI 1-2 [ms]: ", timeint1)             
print("ISI 6-7 [ms]: ", timeint2)             
print("relative difference: ", ISI)
        


#writing to an excel sheet 
import xlwt

from xlwt import Workbook

#create a workbook
wb = Workbook()

#create a a sheet
sheet1 = wb.add_sheet("Master Map Analysis")

sheet1.write(0,0, "Date")
sheet1.write(0,1, "cellchar")
sheet1.write(0,2, "RMP")

for i in range(len(c)):
    sheet1.write(0, 5+i, c[i])
    sheet1.write(1, 5+i, l[i])
                      
wb.save("/mnt/live1/kmikulik/Analysis/test.xlsx")