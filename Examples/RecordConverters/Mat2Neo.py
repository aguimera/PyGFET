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

Units ={'DeltaVgeff':pq.V,
        'Id':pq.A,
        'pH':pq.V}


#==============================================================================
FileName='../../171019/d171019_B10179W11B7_dc8_02.mat'    
#==============================================================================
#FileName='C:\Users\eduard\Dropbox\PyFET\EduScripts\DCInvivo\PythonFiles171010\GrB10179W11-B8-rec9_new.mat'
#
#FileOut='C:\Users\eduard\Dropbox\PyFET\EduScripts\DCInvivo\PythonFiles171010\GrB10179W11-B8-rec9_new.h5'    

FileOr = FileName.split('/')[-1]

out = spio.loadmat(FileName)
for key, value in list(out.items()):
    out[key] = get_matlab_value(value)
    # now you can access the variable(s) you saved in the matlab file as follows
DCData = out['Gr']

out_f = neo.io.NixIO(filename=FileOut)   

out_seg = neo.Segment(name='NewSeg')
out_bl = neo.Block(name='NewBlock')

for Chn,Vals in DCData.iteritems():
    if Chn.startswith('Ch'):
        for Varn,Var in Vals.iteritems():
            if Varn=='tperiod' or Varn=='tstart' or Varn=='Vds' or Varn=='Vgs': continue
            
            sig = neo.AnalogSignal(Var,
                                       units = Units[Varn],
                                       t_start = Vals['tstart']*pq.s,
                                       sampling_rate = (1/Vals['tperiod'])*pq.Hz,
                                       name=Chn+Varn,
                                       file_origin=FileOr)
            out_seg.analogsignals.append(sig)            
    else:
        sig = neo.AnalogSignal(Vals,
                               units = pq.V,
                               t_start = 0*pq.s,
                               sampling_rate = 1*pq.Hz,
                               name=Chn)
        out_seg.analogsignals.append(sig)   


out_seg.analogsignals.append(sig)

out_bl.segments.append(out_seg)

out_f.write_block(out_bl)

out_f.close()


       