#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 18:46:12 2017

@author: aguimera
"""

import platform
import scipy.io as spio
from spyder.utils.iofuncs import get_matlab_value
import neo
import gc
import quantities as pq
import numpy as np

#==============================================================================
# FileName='../170421/Gr170421Rec7.mat'    
#==============================================================================
#FileName='../../GtechData/Ids_14_09_2017_13_05_13_0000.mat'
#FileOut='../../GtechData/RecTest.h5'

FileName = 'C:\\Users\\GAB\\Documents\\GSK\\Rec_b10631W7_22_21_09_2017_15_08_29_0000'
FileOut = 'C:\\Users\\GAB\\Documents\\GSK\\Rec_b10631W7_22_1.h5'



if platform.system() == 'Linux':
    FileOr = FileName.split('/')[-1]
else:
    FileOr = FileName.split(('\\'))[-1]


out = spio.loadmat(FileName)
Data = out['y']
Fs = out['SR\x00'][0,0]

Time = Data[0,:].transpose()[:,0]

if Fs != int(1/(Time[1]-Time[0])):
    print 'Warning FS', Fs, 1/(Time[1]-Time[0])

#A = np.array([])
#for i,t in enumerate(Time[1:]):
#    a = 1/(t-Time[i])
#    A = np.hstack((A, a)) if A.size else a
    


out_f = neo.io.NixIO(filename=FileOut)   
out_seg = neo.Segment(name='NewSeg')
out_bl = neo.Block(name='NewBlock')

for i, dat in enumerate(Data[1:,:,:]):
    val = dat[0,:]
    sig = neo.AnalogSignal(val,
                           units = 'V',
                           t_start = Time[0]*pq.s,
                           sampling_rate = Fs*pq.Hz,
                           name = 'Ch' + str(i),
                           file_origin=FileOr)
    out_seg.analogsignals.append(sig)            

out_bl.segments.append(out_seg)
out_f.write_block(out_bl)
out_f.close()


       