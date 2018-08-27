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
    
def GetDataMat(FileInput):
    TransNames = {'Ch1': 'Ch01',
                  'Ch2': 'Ch02',
                  'Ch3': 'Ch03',
                  'Ch4': 'Ch04',
                  'Ch5': 'Ch05',
                  'Ch6': 'Ch06',
                  'Ch7': 'Ch07',
                  'Ch8': 'Ch08',
                  'Ch9': 'Ch09',
                  'Ch10': 'Ch10',
                  'Ch11': 'Ch11',
                  'Ch12': 'Ch12',
                  'Ch13': 'Ch13',
                  'Ch14': 'Ch14',
                  'Ch15': 'Ch15',
                  'Ch16': 'Ch16'}

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

    FileFilter = r'C:\Users\eduard\Dropbox (GAB GBIO)\GAB GBIO Team Folder\Experimentals\171110\IV\*.mat'
    FileNames = glob.glob(FileFilter)

    for fin in FileNames:
        fin=fin        
        DevDCVals = GetDataMat(fin)
        dd.io.save(fin.replace('.mat', '.h5'), (DevDCVals, ), ('zlib', 1))

##########################################################################
# Gen Vgs neo file
###########################################################################

#VgsNeoOut = '../170608/Rec7Vgs.h5'
#
#TransNames = {'Ch1': 'T01',
#              'Ch2': 'T02',
#              'Ch3': 'T03',
#              'Ch4': 'T04',
#              'Ch5': 'T05',
#              'Ch6': 'T06',
#              'Ch7': 'T07',
#              'Ch8': 'T08',
#              'Ch9': 'T09',
#              'Ch10': 'T10',
#              'Ch11': 'T11',
#              'Ch12': 'T12',
#              'Ch13': 'T13',
#              'Ch14': 'T14',
#              'Ch15': 'T15',
#              'Ch16': 'T16'}



#os.remove(VgsNeoOut)
#out_f = neo.io.NixIO(filename=VgsNeoOut)
#out_seg = neo.Segment(name='NewSeg')
#
#neoVgs = np.array([])
#for v in Vgs:
#    neoVgs = np.hstack((neoVgs, np.ones(152)*v))
#
#sig = neo.AnalogSignal(neoVgs,
#                       units=pq.V,
#                       t_start=0*pq.s,
#                       sampling_rate=1*pq.Hz,
#                       name='Vgs',
#                       file_origin=FileInput)
#out_seg.analogsignals.append(sig)
#
#for chn, dat in DevDCVals.iteritems():
#    neoGM = np.array([])
#    for v in Vgs:
#        gm = np.polyval(dat['GMPoly'][:, 0], v)
#        neoGM = np.hstack((neoGM, np.ones(152)*gm))
#
#    sig = neo.AnalogSignal(neoGM,
#                           units=pq.A/pq.V,
#                           t_start=0*pq.s,
#                           sampling_rate=1*pq.Hz,
#                           name='GM'+chn,
#                           file_origin=FileInput)
#    out_seg.analogsignals.append(sig)
#
#out_bl = neo.Block(name='NewBlock')
#out_bl.segments.append(out_seg)
#out_f.write_block(out_bl)
#out_f.close()

#sig1 = neo.AnalogSignal(neoGM,
#                   units = pq.A/pq.V,
#                   t_start = 0*pq.s,
#                   sampling_rate = 2*pq.Hz,
#                   name='GM'+chn,
#                   file_origin=FileInput)
#
#sig2 = neo.AnalogSignal(neoGM,
#                   units = pq.A/pq.V,
#                   t_start = 0*pq.s,
#                   sampling_rate = 1*pq.Hz,
#                   name='GM'+chn,
#                   file_origin=FileInput)
#
#s = sig1*sig2