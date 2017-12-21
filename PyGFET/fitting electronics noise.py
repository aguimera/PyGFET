# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 10:24:22 2017

@author: RGarcia1
"""

from __future__ import print_function
import numpy as np
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std

import PyFET.PlotDataClass as PlData
import PyFET.AnalyzeData as FetAna
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import interpolate
from NoiseModel_Test_v2 import *

plt.ion()
plt.close('all')    

Rt=DataAC['B10179O9-F3C11-NT4']['Cy000']['Vds'][ivd]/np.polyval(DataAC['B10179O9-F3C11-NT4']['Cy000']['IdsPoly'][:,ivd],Vgs+DataAC['B10179O9-F3C11-NT4']['Cy000']['Ud0'][ivd])
S=(10000+Rt)/(Rch+Rc)*1e-12
  
def Func(s):

    def func(xdata, a):    
        return np.sqrt(np.abs(a)*s)
    return func

popt, pcov=curve_fit(Func(S),Vgs,DataAC['B10179O9-F3C11-NT4']['Cy000']['Irms'],bounds=(0,np.inf),ftol=1e-20,xtol=1e-20,gtol=1e-20)    

axarr3.semilogy(Vgs,DataAC['B10179O9-F3C11-NT4']['Cy000']['Irms'],label='Irms')
axarr3.semilogy(Vgs,np.sqrt(np.abs(popt[0])*S),label='Irms fit')
axarr3.set_ylabel('Irms [A]')
axarr3.set_xlabel('Vgs-Vdirac [V]')
axarr3.set_title('Irms fitting')  
axarr3.legend()