#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 17:40:45 2021

@author: kmikulik
"""

import numpy as np

import matplotlib.pyplot as plt


 
#wir generieren ein Array
xdat= np.arange(0, 1000)


#wir generieren Sinuskurven, die um 0.5*pi verschoben sind
ydat = np.sin( xdat*np.pi/100 )

zdat = np.sin( xdat*np.pi/100 + 0.5*np.pi )


#wenn wir diese Arrays plotten, sehen wir, dass sie verschoben sind
plt.plot(xdat, ydat)
plt.plot(xdat, zdat)
plt.show()

 

xc = np.correlate( ydat, zdat, 'full')


#jetzt schauen wir uns nur den Bereich an, wo die Korrelation am besten ist, also von der Mitte (1000) 
#bis Mitte plus 100 Punkte nach rechts, weil unsere Periode 200 Datenpunkte sind

xc = xc[int(np.round(len(xc)/2)):int(np.round(len(xc)/2)+100)]


#von diesem eingegrenzten Intervall nehmen wir den maximalen Wert
#Der Index von diesem maximalen Wert führt uns dann zur Zeit und damit zur Phasenverschiebung
#je nach Frequenz

mv = np.max(np.abs(xc))

mi = np.where(np.abs(xc)==mv)[0]

 

time_lag = mi

 

plt.plot(xc)