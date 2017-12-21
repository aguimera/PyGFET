# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 17:38:23 2017

@author: RGarcia1
"""
import sys
from PyQt5  import QtWidgets, uic

import numpy as np
import matplotlib.pyplot as plt
import gc
import PyFET.PyFETdb as PyFETdb
import PyFET.DataStructures as PyFETData
import PyFET.AnalyzeData as PyFETAnalyze
import PyFET.PlotData as PyFETPlot
import deepdish as dd
import cmath as cm
#import PyFET.PyFitCheck as FitCk
import xlwt
#############################################################################
# Upload files
#############################################################################
plt.ion()
plt.close('all')

wb = xlwt.Workbook()
ws={}
FminFit=10 #Minimum frequency used for the fitting of the 1/f noise
FmaxFit=100 #Maximum frequency used for the fitting of the 1/f noise
FminInt=5
FmaxInt=5000
#==============================================================================
# vs Freq Excel
#==============================================================================
xAxes='Freq'
yAxes='PSD'
cn='000' #cycle number ex. 000,001...
DeviceName='B10631O1-F5C4-ET1'


ws[DeviceName] = wb.add_sheet(DeviceName)
ws[DeviceName].write(0, 0,xAxes )
ws[DeviceName].write(0, 1,yAxes )
counter=-1

PSD=np.zeros(len(DataAC[DeviceName]['Cy000']['Fpsd']))
for iVgs,Vgs in enumerate(DataAC[DeviceName]['Cy{}'.format(cn)]['Vgs']):

    counter=counter+1
    ws[DeviceName].write(1,1+counter*2,'{}'.format(Vgs))
    Freq=DataAC[DeviceName]['Cy{}'.format(cn)]['Fpsd']

    for iX,X in enumerate(Freq):
        ws[DeviceName].write(2+iX,counter*2,float(X))

        PSD[iX]=DataAC[DeviceName]['Cy000']['PSD']['Vd0'][iVgs,iX]
    for iY,Y in enumerate(PSD):
        ws[DeviceName].write(iY+2,1+counter*2,Y)

wb.save(DeviceName+yAxes+'vs'+xAxes+'{}'.format('.xls'))
