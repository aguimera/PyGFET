#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 12:51:22 2017

@author: aguimera
"""
import matplotlib.colors as mpcolors
import matplotlib.cm as cmx
import PyGFET.DBAnalyze as Dban
from PyGFET.ExportTools import SaveOpenSigures
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import statsmodels.api as sm

plt.close('all')
plt.ion()

CharTable = 'DCcharacts'
AnalyteStep = 'Tromb'
#DeviceNames = ('B10114W2-Xip1N','B10803W17-Xip1S')
#DeviceNames = ('B10803W17-Xip1S', )
#DeviceNames = ('B10114W2-Xip1N', )
#DeviceNames = ('B10803W17-Xip2N', )
DeviceNames = ('B10803W17-Xip6N', )

Conditions = {'Devices.Name=': DeviceNames,
              'CharTable.IsOK>': (0, ),
              'CharTable.IonStrength=': (0.001,),
              'CharTable.FuncStep=': (AnalyteStep, )}

AnalyteConList = Dban.FindCommonValues(Table=CharTable,
                                                 Parameter='CharTable.AnalyteCon',
                                                 Conditions=Conditions)

#%%

Groups = {}
# Fixed Conditions
CondBase = {}
CondBase['Table'] = CharTable
CondBase['Last'] = True
CondBase['Conditions'] = Conditions

for AnalyteCon in AnalyteConList:
    Cgr = CondBase.copy()
    
    Cond = CondBase['Conditions'].copy()
    Cgr['Conditions'] = Cond
    Cond.update({'CharTable.AnalyteCon=': (AnalyteCon, )})
    Groups['{} {}'.format(AnalyteStep, AnalyteCon)] = Cgr




#%%    
    
Vals = Dban.SearchAndGetParam(Groups=Groups,
                                Plot=True,
                                Boxplot=False,
                                Param='Ud0',
                                Vgs=0.1,
                                Ud0Norm=False,
                                Vds=0.05)


for k, v in sorted(Vals.iteritems()):
    print k, 'Ud0 ', np.mean(v), ' +- ',np.std(v)


#%%
    
Conditions = {'Devices.Name=': DeviceNames,
              'CharTable.IsOK>': (0, ),
              'CharTable.IonStrength=': (0.001,),
              'CharTable.FuncStep=': (AnalyteStep, )}

TrtsList = Dban.FindCommonValues(Table=CharTable,
                                           Parameter='Trts.Name',
                                           Conditions=Conditions)

AnalyteConList = Dban.FindCommonValues(Table=CharTable,
                                                 Parameter='CharTable.AnalyteCon',
                                                 Conditions=Conditions)

CondBase = {}
CondBase['Table'] = CharTable
CondBase['Last'] = True
CondBase['Conditions'] = Conditions

plt.figure()
Res = np.array([])
R2 = np.array([])
cmap = cmx.ScalarMappable(mpcolors.Normalize(vmin=0, vmax=len(TrtsList)),
                          cmx.jet)
for itrt, Trt in enumerate(sorted(TrtsList)):
    color = cmap.to_rgba(itrt)
    Conditions.update({'Trts.Name=': (Trt,)})
    ValX = np.array([])
    ValY = np.array([])
    for Conc in sorted(AnalyteConList):
        Conditions.update({'CharTable.AnalyteCon=':(Conc,)})

        vy = Dban.SearchAndGetParam(Groups={'1': CondBase},
                                        Plot=False,
                                        Boxplot=False,
                                        Param='Ud0').values()[0][0,0]
        
        ValY = np.hstack((ValY, vy)) if ValY.size else vy
        ValX = np.hstack((ValX, Conc)) if ValX.size else np.array(Conc)
    
    ValX = np.log10(ValX)
    plt.plot(ValX, ValY, color=color, label=Trt)
    
    X = sm.add_constant(ValX)
    res=sm.OLS(ValY, X).fit()
    R2=np.vstack((R2,res.rsquared)) if R2.size else res.rsquared
    prstd, iv_l, iv_u = wls_prediction_std(res)    
    
    plt.plot(ValX, res.fittedvalues,'k--')
    plt.fill_between(ValX, iv_u, iv_l,
                     color=color,
                     linewidth=0.0,
                     alpha=0.3)
    
    Res = np.vstack((Res,res.params)) if Res.size else res.params

plt.legend()
print Res
print R2
#    plt.plot(ValX,ValY)


