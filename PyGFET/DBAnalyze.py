#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 15:08:05 2017

@author: aguimera
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mpcolors
import matplotlib.cm as cmx
import sys
from itertools import cycle
import statsmodels.api as sm
import xlsxwriter as xlsw

from PyGFET.DataClass import DataCharAC
import PyGFET.DBCore as PyFETdb


def PlotMeanStd(Data, Xvar, Yvar, Vds=None, Ax=None, Ud0Norm=True, Color='r',
                PlotOverlap=False, PlotOverlapMean=False,
                label=None, ScaleFactor=1, **kwargs):

    fontsize = 'medium'
    labelsize = 5
    scilimits = (-2, 2)
    PointsInRange = 100

    if Ax is None:
        fig, Ax = plt.subplots()

    if PlotOverlap:
        for Trtn, Datas in Data.iteritems():
            for Dat in Datas:
                if Dat.IsOK:
                    funcX = Dat.__getattribute__('Get' + Xvar)
                    funcY = Dat.__getattribute__('Get' + Yvar)
                    Valy = funcY(Vds=Vds, Ud0Norm=Ud0Norm) * ScaleFactor
                    Valx = funcX(Vds=Vds, Ud0Norm=Ud0Norm)
                    if Valy is not None:
                        Ax.plot(Valx, Valy, color=Color, alpha=0.2)

    # Search Vgs Vals
    VxMin = []
    VxMax = []
    for Trtn, Datas in Data.iteritems():
        for Dat in Datas:
            if Dat.IsOK:
                funcX = Dat.__getattribute__('Get' + Xvar)
                Valx = funcX(Vds=Vds, Ud0Norm=Ud0Norm)
                if Valx is not None:
                    VxMax.append(np.max(Valx))
                    VxMin.append(np.min(Valx))
    VxMax = np.min(VxMax)
    VxMin = np.max(VxMin)
    ValX = np.linspace(VxMin, VxMax, PointsInRange)
    ValY = np.array([])

    if 'xlsSheet' in kwargs.keys():
        xlscol = 0
        kwargs['xlsSheet'].write(0, xlscol, Xvar + ' - ' + Yvar)
        for ivr, vr in enumerate(ValX):
            kwargs['xlsSheet'].write(ivr+1, xlscol, vr)

    for Trtn, Datas in Data.iteritems():
        for Dat in Datas:
            if Dat.IsOK:
                funcY = Dat.__getattribute__('Get' + Yvar)
                Valy = funcY(Vgs=ValX, Vds=Vds, Ud0Norm=Ud0Norm) * ScaleFactor
                if Valy is not None:
                    ValY = np.hstack((ValY, Valy)) if ValY.size else Valy

                    if 'xlsSheet' in kwargs.keys():
                        xlscol = xlscol + 1
                        kwargs['xlsSheet'].write(0, xlscol, Trtn)
                        for ivr, vr in enumerate(Valy):
                            kwargs['xlsSheet'].write(ivr+1, xlscol, vr)

                    if PlotOverlapMean:
                        plt.plot(ValX, Valy, color=Color, alpha=0.2)

    if ValY.size:
        avg = np.mean(ValY, axis=1)
        std = np.std(ValY, axis=1)
        plt.plot(ValX, avg, color=Color, label=label)
        plt.fill_between(ValX, avg+std, avg-std,
                         color=Color,
                         linewidth=0.0,
                         alpha=0.3)

    if 'xlsSheet' in kwargs.keys():
        xlscol = xlscol + 1
        kwargs['xlsSheet'].write(0, xlscol, 'Avg')
        for ivr, vr in enumerate(avg):
            kwargs['xlsSheet'].write(ivr+1, xlscol, vr)

    if 'xlsSheet' in kwargs.keys():
        xlscol = xlscol + 1
        kwargs['xlsSheet'].write(0, xlscol, 'Std')
        for ivr, vr in enumerate(std):
            kwargs['xlsSheet'].write(ivr+1, xlscol, vr)

    if 'xscale' in kwargs.keys():
        Ax.set_xscale(kwargs['xscale'])
    if 'yscale' in kwargs.keys():
        Ax.set_yscale(kwargs['yscale'])
    Ax.set_ylabel(Yvar, fontsize=fontsize)
    Ax.set_xlabel(Xvar, fontsize=fontsize)
    Ax.tick_params(axis='both', which='Both', labelsize=labelsize)
    Ax.ticklabel_format(axis='y', style='sci', scilimits=scilimits)
    Ax.ticklabel_format(axis='x', style='sci', scilimits=scilimits)


def PlotXYVars(Data, Xvar, Yvar, Vgs, Vds, Ud0Norm=True, label=None,
               Ax=None, Color=None, **kwargs):

    fontsize = 'medium'
    labelsize = 5
    scilimits = (-2, 2)

    if Ax is None:
        fig, Ax = plt.subplots()

    for Trtn, Datas in Data.iteritems():
        for Dat in Datas:
            if Dat.IsOK:
                funcX = Dat.__getattribute__('Get' + Xvar)
                funcY = Dat.__getattribute__('Get' + Yvar)

                try:
                    Valx = funcX(Vgs=Vgs, Vds=Vds, Ud0Norm=Ud0Norm)
                    Valy = funcY(Vgs=Vgs, Vds=Vds, Ud0Norm=Ud0Norm)
                    Ax.plot(Valx, Valy, '*', color=Color, label=label)
                except:  # catch *all* exceptions
                    print Dat.Name, sys.exc_info()[0]

    if 'xscale' in kwargs.keys():
        Ax.set_xscale(kwargs['xscale'])
    if 'yscale' in kwargs.keys():
        Ax.set_yscale(kwargs['yscale'])
#    else:
#        Ax.ticklabel_format(axis='x', style='sci', scilimits=scilimits)

    if 'ylim' in kwargs.keys():
        Ax.set_ylim(kwargs['ylim'])

    Ax.set_ylabel(Yvar, fontsize=fontsize)
    Ax.set_xlabel(Xvar, fontsize=fontsize)
    Ax.tick_params(axis='both', which='Both', labelsize=labelsize)
#    Ax.ticklabel_format(axis='y', style='sci', scilimits=scilimits)


def GetUD0(Data, Vds=None, **kwargs):
    Ud0 = np.array([])
    for Trtn, Datas in Data.iteritems():
        for Dat in Datas:
            ud0 = Dat.GetUd0(Vds)
            if ud0 is not None:
                Ud0 = np.hstack((Ud0, ud0)) if Ud0.size else ud0
    return Ud0


def GetParam(Data, Param, Vgs=None, Vds=None, Ud0Norm=False, **kwargs):
    Vals = np.array([])
    for Trtn, Datas in Data.iteritems():
        for Dat in Datas:
            func = Dat.__getattribute__('Get' + Param)

            try:
                Val = func(Vgs=Vgs, Vds=Vds, Ud0Norm=Ud0Norm)
            except:  # catch *all* exceptions
                print Dat.Name, sys.exc_info()[0]            
                Val = None
    
            if Val is not None:
                Vals = np.hstack((Vals, Val)) if Vals.size else Val
    return Vals


def CheckConditionsCharTable(Conditions, Table):
    for k in Conditions.keys():
        if k.startswith('CharTable'):
            nk = k.replace('CharTable', Table)
            Conditions.update({nk: Conditions[k]})
            del(Conditions[k])
    return Conditions


def FindCommonParametersValues(Parameter, Conditions, Table='ACcharacts'):
    Conditions = CheckConditionsCharTable(Conditions, Table)

    if Parameter.startswith('CharTable'):
        Parameter = Parameter.replace('CharTable', Table)

    MyDb = PyFETdb.PyFETdb(host='opter6.cnm.es',
                           user='pyfet',
                           passwd='p1-f3t17',
                           db='pyFET')
#    MyDb = PyFETdb.PyFETdb()

    Output = (Parameter,)
    Res = MyDb.GetCharactInfo(Table=Table,
                              Conditions=Conditions,
                              Output=Output)

    del (MyDb)
    #  Generate a list of tupples with devices Names and comments
    Values = []
    for Re in Res:
        Values.append(Re[Parameter])

    return set(Values)


def GetFromDB(Conditions, Table='ACcharacts', Last=True, GetGate=True,
              OutilerFilter=None):
    Conditions = CheckConditionsCharTable(Conditions, Table)

    MyDb = PyFETdb.PyFETdb(host='opter6.cnm.es',
                           user='pyfet',
                           passwd='p1-f3t17',
                           db='pyFET')

#    MyDb = PyFETdb.PyFETdb()

    DataD, Trts = MyDb.GetData2(Conditions=Conditions,
                                Table=Table,
                                Last=Last,
                                GetGate=GetGate)

    del(MyDb)

    Data = {}
    for Trtn, Cys in DataD.iteritems():
        print Trtn
        Chars = []
        for Cyn, Dat in Cys.iteritems():
            Char = DataCharAC(Dat)
            Chars.append(Char)
        Data[Trtn] = Chars

    if OutilerFilter is None:
        return Data, Trts

#   Find Outliers
    Vals = np.array([])
    for Trtn, Datas in Data.iteritems():
        for Dat in Datas:
            func = Dat.__getattribute__('Get' + OutilerFilter['Param'])
            Val = func(Vgs=OutilerFilter['Vgs'],
                       Vds=OutilerFilter['Vds'],
                       Ud0Norm=OutilerFilter['Ud0Norm'])
            if Val is not None:
                Vals = np.hstack((Vals, Val)) if Vals.size else Val

    p25 = np.percentile(Vals, 25)
    p75 = np.percentile(Vals, 75)
    lower = p25 - 1.5 * (p75 - p25)
    upper = p75 + 1.5 * (p75 - p25)

    Data = {}
    for Trtn, Cys in DataD.iteritems():
        Chars = []
        for Cyn, Dat in Cys.iteritems():
            Char = DataCharAC(Dat)
            func = Char.__getattribute__('Get' + OutilerFilter['Param'])
            Val = func(Vgs=OutilerFilter['Vgs'],
                       Vds=OutilerFilter['Vds'],
                       Ud0Norm=OutilerFilter['Ud0Norm'])

            if (Val <= lower or Val >= upper):
                print 'Outlier Removed ->', Val, Trtn, Cyn
            else:
                Chars.append(Char)
        Data[Trtn] = Chars

    return Data, Trts


def MultipleSearchParam(Plot=True, Boxplot=False, **kwargs):
    if Plot:
        fig, Ax = plt.subplots()
        xLab = []
        xPos = []

    if 'XlsFile' in kwargs.keys():
        xlswbook = xlsw.Workbook(kwargs['XlsFile'])
        xlssheet = xlswbook.add_worksheet('W1')

    Vals = {}
    Groups = kwargs['Groups']
    for iGr, (Grn, Grc) in enumerate(sorted(Groups.iteritems())):
        print 'Getting data for ', Grn
        Data, Trts = GetFromDB(**Grc)

        if len(Data) > 0:
            vals = GetParam(Data, **kwargs)
            if vals is None:
                continue
            if vals.size == 0:
                continue

            Vals[Grn] = vals

            if 'XlsFile' in kwargs.keys():
                xlssheet.write(0, iGr, Grn)
                for ivr, vr in enumerate(vals[0]):
                    xlssheet.write(ivr+1, iGr, vr)

            if Plot:
                if Boxplot:
                    Ax.boxplot(vals.transpose(), positions=(iGr+1,))
                    xPos.append(iGr+1)
                else:
                    Ax.plot(iGr, vals, '*')
                    xPos.append(iGr)
                xLab.append(Grn)
        else:
            print 'Empty data for ', Grn

    if Plot:
        plt.xticks(xPos, xLab, rotation=45)
        Ax.set_ylabel(kwargs['Param'])
        Ax.grid()
        Ax.ticklabel_format(axis='y', style='sci', scilimits=(2, 2))
        Ax.set_xlim(min(xPos)-0.5, max(xPos)+0.5)
        title = 'Vgs {} Vds {}'.format(kwargs['Vgs'], kwargs['Vds'])
        plt.title(title)
        plt.tight_layout()
        if 'xscale' in kwargs.keys():
            Ax.set_xscale(kwargs['xscale'])
        if 'yscale' in kwargs.keys():
            Ax.set_yscale(kwargs['yscale'])


    if 'XlsFile' in kwargs.keys():
        xlswbook.close()

    return Vals


def CreateCycleColors(Vals):
    ncolors = len(Vals)
    cmap = cmx.ScalarMappable(mpcolors.Normalize(vmin=0, vmax=ncolors),
                              cmx.jet)
    colors = []
    for i in range(ncolors):
        colors.append(cmap.to_rgba(i))

    return cycle(colors)


def MultipleSearch(Func=PlotMeanStd, **kwargs):
    col = CreateCycleColors(kwargs['Groups'])
    fig, Ax = plt.subplots()
    Groups = kwargs['Groups']

    if 'XlsFile' in kwargs.keys():
        xlswbook = xlsw.Workbook(kwargs['XlsFile'])

    for Grn, Grc in sorted(Groups.iteritems()):
        print 'Getting data for ', Grn
        Data, Trts = GetFromDB(**Grc)
        if len(Data) > 0:
            try:
                if 'XlsFile' in kwargs.keys():
                    xlssheet = xlswbook.add_worksheet(Grn)
                    kwargs['xlsSheet'] = xlssheet

                Func(Data,
                     Ax=Ax,
                     Color=col.next(),
                     label=Grn,
                     **kwargs)
            except:
                print Grn, 'ERROR --> ', sys.exc_info()[0]
        else:
            print 'Empty data for ', Grn

    handles, labels = Ax.get_legend_handles_labels()
    hh = []
    ll = []
    for h, l in zip(handles, labels):
        if l not in ll:
            hh.append(h)
            ll.append(l)
    Ax.legend(hh, ll)

    if 'XlsFile' in kwargs.keys():
        xlswbook.close()

    return fig, Ax


def CalcTLM(Groups, Vds=None, Ax=None, Color=None,
            DebugPlot=False, Label=None):
    if Ax is None:
        fig,  AxRs = plt.subplots()
        AxRc = AxRs.twinx()
        fig1,  AxLT = plt.subplots()
    else:
        AxRc = Ax[0]
        AxRs = Ax[1]
        AxLT = Ax[2]

    PointsInRange = 100
    DatV = []
    for Grn, Grc in sorted(Groups.iteritems()):
        print 'Getting data for ', Grn
        Data, Trts = GetFromDB(**Grc)
        DatV.append(Data)

    VxMin = []
    VxMax = []
    for Data in DatV:
        if len(Data) > 0:
            for Trtn, Datas in Data.iteritems():
                for Dat in Datas:
                    funcX = Dat.__getattribute__('GetVgs')
                    Valx = funcX(Vds=Vds, Ud0Norm=True)
                    if Valx is not None:
                        VxMax.append(np.max(Valx))
                        VxMin.append(np.min(Valx))

    VxMax = np.min(VxMax)
    VxMin = np.max(VxMin)
    VGS = np.linspace(VxMin, VxMax, PointsInRange)

    Rsheet = np.ones(VGS.size)*np.NaN
    RsheetMax = np.ones(VGS.size)*np.NaN
    RsheetMin = np.ones(VGS.size)*np.NaN
    Rc = np.ones(VGS.size)*np.NaN
    RcMax = np.ones(VGS.size)*np.NaN
    RcMin = np.ones(VGS.size)*np.NaN
    LT = np.ones(VGS.size)*np.NaN

    if DebugPlot:
        plt.figure()
    Width = None
    for ivg, vgs in enumerate(VGS):
        X = np.array([])
        Y = np.array([])
        for Data in DatV:
            if len(Data) > 0:
                for Trtn, Datas in Data.iteritems():
                    for Dat in Datas:
                        rds = Dat.GetRds(Vgs=vgs, Vds=Vds, Ud0Norm=True)
                        Y = np.vstack((Y, rds)) if Y.size else rds
                        L = np.array((Dat.TrtTypes['Length']))
                        X = np.vstack((X, L)) if X.size else L
                        if Width is None:
                            Width = Dat.TrtTypes['Width']
                        else:
                            if not Width == Dat.TrtTypes['Width']:
                                print Trtn, 'WARNING Bad width'
        if DebugPlot:
            plt.plot(X, Y, '*')

        X = sm.add_constant(X)
        res = sm.OLS(Y, X).fit()
        Rsheet[ivg] = res.params[1] * Width
        RsheetMax[ivg] = (res.bse[1]+res.params[1]) * Width
        RsheetMin[ivg] = (-res.bse[1]+res.params[1]) * Width
        Rc[ivg] = res.params[0]
        RcMax[ivg] = res.bse[0]+res.params[0]
        RcMin[ivg] = -res.bse[0]+res.params[0]
        LT[ivg] = (res.params[0]/res.params[1])/2

    AxRc.plot(VGS, Rc, color=Color, label=Label)
    AxRc.fill_between(VGS, RcMax, RcMin,
                      color=Color,
                      linewidth=0.0,
                      alpha=0.3)
    AxRs.plot(VGS, Rsheet, '--', color=Color)
    AxRs.fill_between(VGS, RsheetMax, RsheetMin,
                      color=Color,
                      linewidth=0.0,
                      alpha=0.3)

    AxLT.plot(VGS, LT, color=Color)

    AxRc.set_ylabel('Rc')
    AxRs.set_ylabel('Rsheet')
    AxRc.legend()

    ContactVals = {}
    ContactVals['Rsheet'] = Rsheet
    ContactVals['RsheetMax'] = RsheetMax
    ContactVals['RsheetMin'] = RsheetMin
    ContactVals['Rc'] = Rc
    ContactVals['RcMax'] = RcMax
    ContactVals['RcMin'] = RcMin
    ContactVals['VGS'] = VGS
    ContactVals['LT'] = LT

    return ContactVals


#==============================================================================
#     AxRc.plot(VGS, Rc, color=Color, label=Label)
#     AxRc.fill_between(VGS, RcMax, RcMin,
#                       color=Color,
#                       linewidth=0.0,
#                       alpha=0.3)
#     AxRs.plot(VGS, Rsheet, '--', color=Color)
#     AxRs.fill_between(VGS, RsheetMax, RsheetMin,
#                       color=Color,
#                       linewidth=0.0,
#                       alpha=0.3)
##==============================================================================
#
#
#    AxRc.set_ylabel('Rc')
#    AxRs.set_ylabel('Rsheet')
#    AxRc.legend()