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
import matplotlib.pyplot as plt

import h5py
#==============================================================================
# FileName='../170421/Gr170421Rec7.mat'    
#==============================================================================
FileName='../../GtechData/Ids_vs_Vgs_1.mat'



if platform.system() == 'Linux':
    FileOr = FileName.split('/')[-1]
else:
    print 'Windows'


out = h5py.File(FileName)

Data = np.array(out['y'])



plt.plot(Data[:,17],Data[:,2])
       