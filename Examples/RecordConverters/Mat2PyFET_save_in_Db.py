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
import neo
import numpy as np
import quantities as pq
import os
import deepdish as dd




FileInput = '../170608/B9872W24B1dcinitial01_2Vds500.mat'
DCFileOut = '../170608/B9872W24B1dcinitial01_2Vds500.h5'

Time = datetime.datetime.fromtimestamp(os.path.getctime(FileInput))

MatCon = sio.loadmat(FileInput)
Vdc = MatCon['Vdc']
VdcChan = Vdc[0, 0]

DevDCVals = {}

for ch in VdcChan.dtype.names:
    Vgs = VdcChan[ch]['Vgs'][0, 0].transpose()[:, 0]
    Ids = VdcChan[ch]['Ids'][0, 0].transpose()
    Vds = VdcChan[ch]['Vds'][0, 0].transpose()[0]

    DCVals = {'Ids': Ids,
              'Vds': Vds,
              'Vgs': Vgs,
              'ChName': ch,
              'Name': ch,
              'DateTime': Time}
    DevDCVals[ch] = DCVals

FETana.CalcGM(DevDCVals)
FETana.CheckIsOK(DevDCVals, RdsRange=[400, 10e3])

pltDC = FETplt.PyFETPlot()
pltDC.AddAxes(('Ids', 'Gm', 'Rds'))
pltDC.PlotDataCh(DevDCVals)
pltDC.AddLegend()

dd.io.save(DCFileOut, (DevDCVals, ), ('zlib', 1))

##########################################################################
# Gen Vgs neo file
###########################################################################

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