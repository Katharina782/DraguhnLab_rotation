                                    #   %reset -f
# Import libraries.
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datasheet_mod_v01 import datasheet_mod_v01
#mport pingouin as pg
#rom pingouin import pairwise_corr, read_dataset
sns.set(style="ticks", palette='rocket')
from matplotlib import rc
rc("pdf", fonttype=42)
#from sinaplot import  sinaplot
def cm2inch(value):
    return value/2.54


    
efile1 = '/mnt/archive1/cthome/Backup_ProjectArchive/Y_AISdiversity_PYR/Kathi/' +\
    '23_02_2021/MKcellchar_021.cfs'
    
    #%% Niko code
    
    
import stfio
import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib
import pandas as pd
import openpyxl as op
from scipy.signal import find_peaks
from scipy import stats
import math
#from scipy.optimize import curve_fit
import os

# rec = stfio.read(file_name,"tcfs")
# rec=np.array(rec)
# traces = len(rec[1]) 
# data_raw=np.empty((0,8),float)
    
#%%
rec1 = stfio.read(efile1,"tcfs")
rec2=np.array(rec1)
timeline = np.arange(0,2000,0.02).tolist()
#ntraces = len(rec2[1])


#len(timeline)
#data_raw = np.empty((0,8),float)


x= rec2[1][30]
peaks, _ = find_peaks(x,height=0)
plt.plot(x)
plt.plot(peaks, x[peaks],'x')
plt.plot(np.zeros_like(x),'--',color='gray')