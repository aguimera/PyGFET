#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 12:51:22 2017

@author: aguimera
"""

import PyGFET.DBAnalyze as Dban
from PyGFET.ExportTools import SaveOpenSigures
import matplotlib.pyplot as plt
import numpy as np

#%% Find conditions in DB

plt.close('all')
plt.ion()

CharTable = 'DCcharacts'
AnalyteStep = 'Tromb'


#DeviceNames = 'B9355O26-F5C5'
DeviceNames = ('B10114W2-Xip1N','B10803W17-Xip1S')
#DeviceNames = ('B10803W17-Xip1S')

Conditions = {'Devices.Name=': DeviceNames,
              'CharTable.IsOK>': (0, )}

FuncStepList = Dban.FindCommonParametersValues(Parameter='CharTable.FuncStep',
                                               Conditions=Conditions,Table=CharTable)

Conditions['CharTable.FuncStep='] = (AnalyteStep,)
AnalyteConList = Dban.FindCommonParametersValues(Table=CharTable,
                                                 Parameter='CharTable.AnalyteCon',
                                                 Conditions=Conditions)

Groups = {}
# Fixed Conditions
CondBase = {}
CondBase['Table'] = CharTable
CondBase['Last'] = True
CondBase['Conditions'] = Conditions


#%% Genarate Analysis Groups

# Iter conditions
for FuncStep in FuncStepList:
    if FuncStep == AnalyteStep:
        for AnalyteCon in AnalyteConList:
            Cgr = CondBase.copy()
            Cond = CondBase['Conditions'].copy()
            Cgr['Conditions'] = Cond
            Cond.update({'CharTable.FuncStep=': (FuncStep,)})
            Cond.update({'CharTable.AnalyteCon=': (AnalyteCon, )})
            Groups['{} {}'.format(FuncStep, AnalyteCon)] = Cgr
    else:
        Cgr = CondBase.copy()
        
        Cond = CondBase['Conditions'].copy()
        Cgr['Conditions'] = Cond
        Cond.update({'CharTable.FuncStep=': (FuncStep,)})
        Groups[FuncStep] = Cgr


#%%

Dban.MultipleSearch(Groups=Groups,
                    Xvar='Vgs',
                    Yvar='Ids',
                    PlotOverlap=True,
                    Ud0Norm=False)

Dban.MultipleSearch(Groups=Groups,
                    Xvar='Vgs',
                    Yvar='GM',
                    PlotOverlap=True,
                    Ud0Norm=False)

Vals = Dban.MultipleSearchParam(Groups=Groups,
                                Plot=True,
                                Boxplot=True,
                                Param='Ud0',
                                Vgs=0.1,
                                Ud0Norm=False,
                                Vds=0.1)

Vals = Dban.MultipleSearchParam(Groups=Groups,
                                Plot=True,
                                Boxplot=True,
                                Param='Ud0',
                                Vgs=0.1,
                                Ud0Norm=False,
                                Vds=0.1)

Vals = Dban.MultipleSearchParam(Groups=Groups,
                                Plot=True,
                                Boxplot=True,
                                Param='GM',
                                Vgs=-0.1,
                                Ud0Norm=True,
                                Vds=0.1)

#
#MultipleSearch(Groups=Groups,
#               Xvar='Vgs',
#               Yvar='GM',
#               PlotOverlap=True,
#               Ud0Norm=True)
#
#MultipleSearch(Func=PlotXYVars,
#               Groups=Groups,
#               Xvar='Ids',
#               Yvar='GM',
#               Vgs=-0.1,
#               Ud0Norm=True,
#               Vds=0.05)
#
#Vals = MultipleSearchParam(Groups=Groups,
#                           Plot=True,
#                           Boxplot=True,
#                           Param='Ud0',
#                           Vgs=0.1,
#                           Ud0Norm=False,
#                           Vds=0.05)
##
for k, v in sorted(Vals.iteritems()):
    print k, 'Ud0 ', np.mean(v), ' +- ',np.std(v)
#
#Vals = MultipleSearchParam(Groups=Groups,
#                           Plot=True,
#                           Param='GM',
#                           Vgs=-0.1,
#                           Ud0Norm=True,
#                           Vds=0.05)
