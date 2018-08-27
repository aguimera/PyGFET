#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 11:03:16 2017

@author: aguimera
"""

import PyFET.PlotDataClass as FETplt
import PyFET.AnalyzeData as FETana
import PyFET.DataStructures as FETdata
import scipy.io as sio
import datetime, os.path
import numpy as np
import quantities as pq
import os
import deepdish as dd
import glob
    

    MatCon = sio.loadmat(FileInput)
    Vdc = MatCon['Vdc']
    VdcChan = Vdc[0, 0]

    Time = datetime.datetime.fromtimestamp(os.path.getmtime(FileInput))

    DevDCVals = {}
    for ch in VdcChan.dtype.names:
        Vgs = VdcChan[ch]['Vgs'][0, 0].transpose()[:, 0]
        Ids = VdcChan[ch]['Ids'][0, 0].transpose()
        Vds = VdcChan[ch]['Vds'][0, 0].transpose()[0]

        DCVals = {'Ids': Ids,
                  'Vds': Vds,
                  'Vgs': Vgs,
                  'ChName': TransNames[ch],
                  'Name': TransNames[ch],
                  'DateTime': Time}
        DevDCVals[TransNames[ch]] = DCVals

    FETana.CalcGM(DevDCVals)
    FETana.CheckIsOK(DevDCVals, RdsRange=[400, 10e3])

    pltDC = FETplt.PyFETPlot()
    pltDC.AddAxes(('Ids', 'Gm', 'Rds'))
    pltDC.PlotDataCh(DevDCVals)
    pltDC.AddLegend()

    return DevDCVals

if __name__ == "__main__":

#FileInput = '../170608/B9872W24B1dcinitial01_2Vds500.mat'
#DCFileOut = '../170608/B9872W24B1dcinitial01_2Vds500.h5'

#==============================================================================
#     FileFilter = '/home/aguimera/UserGuimeraLocal/SGFETs/Experimentals/171031/*.mat'
#==============================================================================
    FileFilter = r'C:\Users\eduard\Dropbox (GAB GBIO)\GAB GBIO Team Folder\Experimentals\171010\IV\*.mat'
    FileNames = glob.glob(FileFilter)

    for fin in FileNames:        
        DevDCVals = GetDataMat(fin)
        dd.io.save(fin.replace('.mat', '.h5'), (DevDCVals, ), ('zlib', 1))



