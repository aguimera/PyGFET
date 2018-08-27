#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 18:46:12 2017

@author: aguimera
"""

import scipy.io as spio
from spyder.utils.iofuncs import get_matlab_value
import neo
import gc
import quantities as pq
import glob

FileFiler ='../../171010/Rec9/*.mat.mat'

FileNames = glob.glob(FileFiler)

for FileName in FileNames:
    out = spio.loadmat(FileName)
    for key, value in list(out.items()):
        out[key] = get_matlab_value(value)
    Data = out['out'].transpose()
    ChNames = out['outh']
    
    DCData={}
    for chn, dat in zip(ChNames, Data):
        DCData[chn] = dat
    
    out_f = neo.io.NixIO(filename=FileName.replace('.mat.mat','.h5').replace('\\','/'))
    out_seg = neo.Segment(name='NewSeg')
    out_bl = neo.Block(name='NewBlock')
    
    for Chn,Vals in DCData.iteritems():
        if Chn == 'Temps' or Chn == 'pH':
            continue
    
        sig = neo.AnalogSignal(Vals,
                               units = pq.V,
                               t_start = 0*pq.s,
                               sampling_rate = 1*pq.Hz,
                               name=str(Chn))
        out_seg.analogsignals.append(sig)   
    
    out_seg.analogsignals.append(sig)
    out_bl.segments.append(out_seg)
    out_f.write_block(out_bl)
    out_f.close()


       