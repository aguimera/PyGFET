#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 12:51:22 2017

@author: aguimera
"""
import PyGFET.PyFETdb as PyFETdb
from PyGFET.PyFETDataClass import PyFETPlotDataClass as PyFETplt
import PyGFET.PyFETdbAnalyze as Dban
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib import cm
import matplotlib.colors as colors
import xlsxwriter
import numpy as np
import datetime
import tempfile
import shutil
import sys
from itertools import cycle

Cortical16Map = {'Shape': (4, 4),
                 'Ch01': (2, 3),
                 'Ch02': (1, 2),
                 'Ch03': (3, 3),
                 'Ch04': (0, 2),
                 'Ch05': (0, 3),
                 'Ch06': (3, 2),
                 'Ch07': (1, 3),
                 'Ch08': (2, 2),
                 'Ch09': (2, 0),
                 'Ch10': (1, 1),
                 'Ch11': (3, 0),
                 'Ch12': (0, 1),
                 'Ch13': (0, 0),
                 'Ch14': (3, 1),
                 'Ch15': (1, 0),
                 'Ch16': (2, 1)}


def GetCycleColors(nColors, CMap=cm.jet):
    cmap = cm.ScalarMappable(colors.Normalize(vmin=0, vmax=nColors), CMap)
    col = []
    for i in range(nColors):
        col.append(cmap.to_rgba(i))
    return cycle(col)


def PlotXYLine(Data, Xvar, Yvar, Vgs, Vds, Ud0Norm=True, label=None,
               Ax=None, Color=None, **kwargs):

    fontsize = 'medium'
    labelsize = 5
    scilimits = (-2, 2)

    if Ax is None:
        fig, Ax = plt.subplots()

    for Trtn, Datas in Data.iteritems():
        ValX = np.array([])
        ValY = np.array([])
        for Dat in Datas:
            if Dat.IsOK:
                funcX = Dat.__getattribute__('Get' + Xvar)
                funcY = Dat.__getattribute__('Get' + Yvar)
                try:
                    Valx = funcX(Vgs=Vgs, Vds=Vds, Ud0Norm=Ud0Norm)
                    Valy = funcY(Vgs=Vgs, Vds=Vds, Ud0Norm=Ud0Norm)
                    ValX = np.vstack((ValX, Valx)) if ValX.size else Valx
                    ValY = np.vstack((ValY, Valy)) if ValY.size else Valy
#                    Ax.plot(Valx, Valy, '*', color=Color, label=label)
                except:  # catch *all* exceptions
                    print Dat.Name, sys.exc_info()[0]

        try:    
            if ValX.size == 0:
                continue
            SortIndex = np.argsort(ValX, axis=0)
    #        Ax.plot(ValY[SortIndex][:,:,0], color=Color, label=label)
            Ax.plot(ValX[SortIndex][:,:,0], ValY[SortIndex][:,:,0], color=Color, label=label)       
        except:
            print Trtn, sys.exc_info()[0]

    if 'xscale' in kwargs.keys():
        Ax.set_xscale(kwargs['xscale'])
    if 'yscale' in kwargs.keys():
        Ax.set_yscale(kwargs['yscale'])
    else:
        Ax.ticklabel_format(axis='y', style='sci', scilimits=scilimits)

    if 'ylim' in kwargs.keys():
        Ax.set_ylim(kwargs['ylim'])

    Ax.set_ylabel(Yvar, fontsize=fontsize)
    Ax.set_xlabel(Xvar, fontsize=fontsize)
    Ax.tick_params(axis='both', which='Both', labelsize=labelsize)


def CalcParMap(Data, ParMap, ParArgs, ProbeMap):
    Map = np.zeros(ProbeMap['Shape'])*np.NaN
    ret = False
    for Trtn, Dat in Data.iteritems():
        if not Dat[0].IsOK:
            continue
        ch = Trtn.split('-')[-1]
        func = Dat[0].__getattribute__('Get' + ParMap)
        val = func(**ParArgs)
        if val is None:
            continue
        if val.size == 0:
            continue
        if ch not in ProbeMap:
            continue
        Map[ProbeMap[ch]] = val
        ret = True
    if ret:
        return Map
    else:
        return None


class GenXlsDeviceHistory():
    DtMax = np.timedelta64(1, 's')
    FigsDpi = 150  # Resolution for figures

    MeasDCFields = {'Time': ('Meas Date', 0, {}),
                    'Vds': ('Vds', 1, {'Vgs': -0.1,
                                       'Vds': None,
                                       'Ud0Norm': True}),
                    'Ud0': ('Ud0', 2, {'Vgs': -0.1,
                                       'Vds': None,
                                       'Ud0Norm': True}),
                    'Rds': ('Rds', 3, {'Vgs': -0.1,
                                       'Vds': None,
                                       'Ud0Norm': True}),
                    'GM': ('GM', 4, {'Vgs': -0.1,
                                     'Vds': None,
                                     'Ud0Norm': True})}
    MeasACFields = {'Time': ('Meas Date', 0, {}),
                    'Vrms': ('Vrms', 1, {'Vgs': -0.1,
                                         'Vds': None,
                                         'Ud0Norm': True})}

    DCTimePlotPars = ('Ids', 'Rds', 'GM', 'Ig')
    ACTimePlotPars = ('Ids', 'GM', 'Vrms','Irms')

    TimePars = ('Ud0', 'Rds', 'Ids', 'GM', 'Vrms')
    TimeParsArgs = ({'Vgs': None, 'Vds': None},
                    {'Vgs': -0.1, 'Vds': None, 'Ud0Norm': True, 'ylim': (500, 10e3)},
                    {'Vgs': -0.1, 'Vds': None, 'Ud0Norm': True},
                    {'Vgs': -0.1, 'Vds': None, 'Ud0Norm': True, 'ylim': (-5e-4, 0)},
                    {'Vgs': -0.1, 'Vds': None, 'Ud0Norm': True, 'yscale':'log', 'ylim': (1e-5, 2e-4)})

    TrtInfoFields = {'VTrts.DCMeas': ('DCMeas', 0, 0),
                     'VTrts.ACMeas': ('ACMeas', 1, 0),
                     'VTrts.GMeas': ('GMeas', 2, 0),
                     'TrtTypes.Name': ('Trt Type', 3, 0),
                     'TrtTypes.Length': ('Lenght', 4, 0),
                     'TrtTypes.Width': ('Width', 5, 0),
                     'TrtTypes.Pass': ('Pass', 6, 0),
                     'TrtTypes.Area': ('Area', 7, 0),
                     'TrtTypes.Contact': ('Contact', 8, 0),
                     'Trts.Comments': ('T-Comments', 9, 0)}

    DevInfoFields = {'Devices.Name': ('Device', 0, 0),
                     'Devices.Comments': ('D-Comments', 1, 0),
                     'Devices.State': ('D-State', 2, 0),
                     'Devices.ExpOK': ('D-ExpOK', 3, 0),
                     'Wafers.Masks': ('W-Masks', 4, 0),
                     'Wafers.Graphene': ('W-Graphene', 5, 0),
                     'Wafers.Comments': ('W-Comments', 6, 0)}

    def GetSortData(self, TrtName):
        Conditions = {'Trts.Name=': (TrtName, )}
        CharTable = 'DCcharacts'
        DatDC, _ = Dban.GetFromDB(Conditions=Conditions,
                                  Table=CharTable,
                                  Last=False,
                                  GetGate=True)
        CharTable = 'ACcharacts'
        DatAC, _ = Dban.GetFromDB(Conditions=Conditions,
                                  Table=CharTable,
                                  Last=False,
                                  GetGate=True)

        DataDC = DatDC.values()[0]
        if len(DatAC.values()) == 0:
            DataAC = []
        else:
            DataAC = DatAC.values()[0]

        DCtimes = []
        for dat in DataDC:
            DCtimes.append(dat.GetTime()[0, 0])
        ACtimes = []
        for dat in DataAC:
            ACtimes.append(dat.GetTime()[0, 0])
        SortList = []
        for idct, DCt in enumerate(DCtimes):
            idacdc = None
            for iact, ACt in enumerate(ACtimes):
                if np.abs(DCt - ACt) < self.DtMax:
                    idacdc = iact
            SortList.append((DCt, idct, idacdc))
        SortList.sort()

        self.DictDC = DatDC
        self.DictAC = DatAC
        self.DataDC = DataDC
        self.DataAC = DataAC
        self.SortList = SortList
        self.TimeParsDatdict = (self.DictDC,  # TODO fix this dictionary
                                self.DictDC,
                                self.DictDC,
                                self.DictDC,
                                self.DictAC)

    def __init__(self, FileName, Conditions):
        self.dbConditions = Conditions
        GroupBy = 'Trts.Name'
        self.TrtsList = Dban.FindCommonParametersValues(Table='DCcharacts',
                                                        Parameter=GroupBy,
                                                        Conditions=Conditions)
        GroupBy = 'Devices.Name'
        self.DeviceName = Dban.FindCommonParametersValues(Table='DCcharacts',
                                                          Parameter=GroupBy,
                                                          Conditions=Conditions)

        self.WorkBook = xlsxwriter.Workbook(FileName)
        self.WorkBook.add_format()

        self.Fbold = self.WorkBook.add_format({'bold': True})
        self.FOK = self.WorkBook.add_format({'num_format': '###.00E+2',
                                             'font_color': 'black'})
        self.FNOK = self.WorkBook.add_format({'num_format': '###.00E+2',
                                              'font_color': 'red'})
        self.TmpPath = tempfile.mkdtemp(suffix='PyFET')

# Init Db connection
        self.Mydb = PyFETdb.PyFETdb(host='opter6.cnm.es',
                                    user='pyfet',
                                    passwd='p1-f3t17',
                                    db='pyFET')

        self.WorkBook.add_worksheet('Summary') 
        for TrtName in sorted(self.TrtsList):
            self.WorkBook.add_worksheet(TrtName)

    def GenFullReport(self):
        for TrtName in self.TrtsList:
            self.GenTrtReport(TrtName)
            plt.close('all')

        Sheet = self.WorkBook.sheetnames['Summary']
        for v in self.DevInfoFields.values():
            Sheet.write(v[1], 0, v[0], self.Fbold)
        Tinf = self.Mydb.GetDevicesInfo(Conditions={'Devices.Name=': self.DeviceName},
                                        Output=self.DevInfoFields.keys())
        for k, val in Tinf[0].iteritems():
            row = self.DevInfoFields[k][1]
            col = 1
            Sheet.write(row, col, val)

        Fig, Ax = plt.subplots(len(self.TimePars), 1, sharex=True, figsize=(12, 12))
        ColCy = GetCycleColors(len(self.TrtsList))
        for TrtName in self.TrtsList:
            self.GetSortData(TrtName)
            TrtColor = ColCy.next()
            for iPar, Par in enumerate(self.TimePars):
                PlotXYLine(Data=self.TimeParsDatdict[iPar],
                           Xvar='Time',
                           Yvar=Par,
                           Ax=Ax[iPar],
                           Color=TrtColor,
                           **self.TimeParsArgs[iPar])
        plt.tight_layout()
        plt.subplots_adjust(hspace=0)
        fname = tempfile.mktemp(suffix='.png', dir=self.TmpPath)
        Fig.savefig(fname, dpi=self.FigsDpi)
        Sheet.insert_image(len(self.DevInfoFields)+2, 0, fname)

    def GenTrtReport(self, TrtName):
        Sheet = self.WorkBook.sheetnames[TrtName]
        Sheet.write(0, 0, 'Trt Name', self.Fbold)
        Sheet.write(0, 1, TrtName)

        for v in self.TrtInfoFields.values():
            Sheet.write(v[1], 0, v[0], self.Fbold)
        Tinf = self.Mydb.GetTrtsInfo(Conditions={'Trts.Name=': (TrtName, )},
                                     Output=self.TrtInfoFields.keys())
        for k, val in Tinf[0].iteritems():
            row = self.TrtInfoFields[k][1]
            col = 1
            Sheet.write(row, col, val)

        self.GetSortData(TrtName)
        self.FillHistoryTable(Sheet, Loc=(len(self.TrtInfoFields)+2, 0))
# Insert DC time evolution plots
        col = len(self.MeasDCFields) + len(self.MeasACFields) + 1
        Plot = PyFETplt()
        Plot.AddAxes(self.DCTimePlotPars)
        Plot.PlotDataSet(DataDict=self.DictDC,
                         Trts=self.DictDC.keys(),
                         ColorOn='Date',
                         MarkOn=None)
        fname = tempfile.mktemp(suffix='.png', dir=self.TmpPath)
        Plot.Fig.savefig(fname, dpi=self.FigsDpi)
        Sheet.insert_image(0, 10, fname)
# Insert AC time evolution plots
        Plot = PyFETplt()
        Plot.AddAxes(self.ACTimePlotPars)
        Plot.PlotDataSet(DataDict=self.DictAC,
                         Trts=self.DictAC.keys(),
                         ColorOn='Date',
                         MarkOn=None)
        fname = tempfile.mktemp(suffix='.png', dir=self.TmpPath)
        Plot.Fig.savefig(fname, dpi=self.FigsDpi)
        Sheet.insert_image(30, 10, fname)
# Insert time evolution parameters
        Fig, Ax = plt.subplots(len(self.TimePars), 1, sharex=True)
        for iPar, Par in enumerate(self.TimePars):
            Dban.PlotXYVars(Data=self.TimeParsDatdict[iPar],
                            Xvar='DateTime',
                            Yvar=Par,
                            Ax=Ax[iPar],
                            **self.TimeParsArgs[iPar])
        plt.tight_layout()
        plt.subplots_adjust(hspace=0)
        fname = tempfile.mktemp(suffix='.png', dir=self.TmpPath)
        Fig.savefig(fname, dpi=self.FigsDpi)
        Sheet.insert_image(60, 10, fname)

    def FillHistoryTable(self, Sheet, Loc):
        RowOff = Loc[0]
        ColOff = Loc[1]

        for k, v in self.MeasDCFields.iteritems():
            Sheet.write(RowOff, ColOff+v[1], v[0], self.Fbold)
        coloff = ColOff + len(self.MeasDCFields)
        for k, v in self.MeasACFields.iteritems():
            Sheet.write(RowOff, coloff+v[1], v[0], self.Fbold)

        for iMea, SortInd in enumerate(self.SortList):
            Row = iMea + RowOff + 1
            DCi = SortInd[1]
            ACi = SortInd[2]

            if self.DataDC[DCi].IsOK:
                Format = self.FOK
            else:
                Format = self.FNOK

            for par, v in self.MeasDCFields.iteritems():
                Col = ColOff + v[1]
                func = self.DataDC[DCi].__getattribute__('Get' + par)
                val = func(**v[2])
                if val is None:
                    continue
                if hasattr(val, '__iter__'):
                    if val.size == 0:
                        continue
                if par == 'Time':
                    val = val[0, 0].astype(datetime.datetime).strftime('%x %X')
                    Sheet.set_column(Col, Col, width=20)
                Sheet.write(Row, Col, val, Format)

            if ACi is None:
                continue

            coloff = ColOff + len(self.MeasDCFields)
            for par, v in self.MeasACFields.iteritems():
                Col = coloff + v[1]
                func = self.DataAC[ACi].__getattribute__('Get' + par)
                val = func(**v[2])
                if val is None:
                    continue
                if hasattr(val, '__iter__'):
                    if val.size == 0:
                        continue
                if par == 'Time':
                    val = val[0, 0].astype(datetime.datetime).strftime('%x %X')
                    Sheet.set_column(Col, Col, width=20)
                Sheet.write(Row, Col, val, Format)

    def close(self):
        self.WorkBook.close()
        shutil.rmtree(self.TmpPath)


class GenXlsReport():
     # ('IdTrt',0,0) (Header, position, string Conv)
    DevInfoFields = {'Devices.Name': ('Device', 0, 0),
                     'Devices.Comments': ('D-Comments', 1, 0),
                     'Devices.State': ('D-State', 2, 0),
                     'Devices.ExpOK': ('D-ExpOK', 3, 0),
                     'Wafers.Masks': ('W-Masks', 4, 0),
                     'Wafers.Graphene': ('W-Graphene', 5, 0),
                     'Wafers.Comments': ('W-Comments', 6, 0)}

    # ('IdTrt',0,0) (Header, position, CountOkDevices() Parameters)
    DevOKFields = {'IsOK': ('Working', 0, {'RefVal': None,
                                           'Lower': None,
                                           'ParArgs': None}),
                   'Rds': ('Rds', 1, {'RefVal': 5e3,
                                      'Lower': True,
                                      'ParArgs': {'Vgs': -0.1,
                                                  'Vds': None,
                                                  'Ud0Norm': True}}),
                   'Vrms': ('Vrms', 2, {'RefVal': 100e-6,
                                        'Lower': True,
                                        'ParArgs': {'Vgs': -0.1,
                                                    'Vds': None,
                                                    'Ud0Norm': True}})}

    MeasInfoTableLoc = (15, 0)
    # ('IdTrt',0,0) (Header, position, string Conv)
    TrtInfoFields = {'VTrts.DCMeas': ('DCMeas', 0, 0),
                     'VTrts.ACMeas': ('ACMeas', 1, 0),
                     'VTrts.GMeas': ('GMeas', 2, 0),
                     'TrtTypes.Name': ('Trt Type', 3, 0),
                     'TrtTypes.Length': ('Lenght', 4, 0),
                     'TrtTypes.Width': ('Width', 5, 0),
                     'TrtTypes.Pass': ('Pass', 6, 0),
                     'TrtTypes.Area': ('Area', 7, 0),
                     'TrtTypes.Contact': ('Contact', 8, 0),
                     'Trts.Comments': ('T-Comments', 9, 0)}

    # ('IdTrt',0,0) (Header, position, CalcParameters)
    MeasFields = {'Time': ('Meas Date', 0, {}),
                  'Vds': ('Vds', 1, {'Vgs': -0.1,
                                     'Vds': None,
                                     'Ud0Norm': True}),
                  'Ud0': ('Ud0', 2, {'Vgs': -0.1,
                                     'Vds': None,
                                     'Ud0Norm': True}),
                  'Rds': ('Rds', 3, {'Vgs': -0.1,
                                     'Vds': None,
                                     'Ud0Norm': True}),
                  'GM': ('GM', 4, {'Vgs': -0.1,
                                   'Vds': None,
                                   'Ud0Norm': True}),
                  'Vrms': ('Vrms', 5, {'Vgs': -0.1,
                                       'Vds': None,
                                       'Ud0Norm': True})}

    FigsDpi = 150  # Resolution for figures

    ProbeMap = Cortical16Map
    MapLoc = (0, 4)
    MapsFigSize = (3, 3)
    MapSpacing = 6
    MapPars = ('Rds', 'Vrms')
    MapParArgs = ({'Vgs': -0.1, 'Vds': None, 'Ud0Norm': True},
                  {'Vgs': -0.1, 'Vds': None, 'Ud0Norm': True})
    MapNorm = (colors.Normalize(200, 1e4),
               colors.LogNorm(1e-5, 1e-4))
    MapUnits = ('Ohms', 'Vrms')

    CharFigLoc = (None, 0)  # If none after Electchar
    CharPars = ('Ids', 'GM', 'Vrms', 'Rds')
    CharParLeg = 'Rds'

    SummaryPlotSpacing = 25
    SummaryLinePlots =({'Xvar': 'Vgs', 'Yvar': 'Ids', 'Vds':None, 'PlotOverlap': False, 'Ud0Norm':False},
                       {'Xvar': 'Vgs', 'Yvar': 'GM', 'Vds':None, 'PlotOverlap': False, 'Ud0Norm':False},
                       {'Xvar': 'Vgs', 'Yvar': 'Vrms', 'Vds':None, 'PlotOverlap': False, 'Ud0Norm':True, 'yscale':'log'})

    SummaryBoxPlots =({'Plot': True, 'Boxplot': False, 'Param': 'Vrms', 'Vgs': -0.1, 'Ud0Norm': True, 'Vds':None, 'yscale':'log'},
                      {'Plot': True, 'Boxplot': True, 'Param': 'Ud0', 'Vgs': -0.1, 'Ud0Norm': True, 'Vds':None})

    def __init__(self, FileName, Conditions, PathHistory=None):
# Init WorkBook, formats and temp folder
        self.WorkBook = xlsxwriter.Workbook(FileName)
        self.FYeild = self.WorkBook.add_format({'num_format': '0.00%',
                                                'font_color': 'blue',
                                                'bold': True})
        self.Fbold = self.WorkBook.add_format({'bold': True})
        self.FOK = self.WorkBook.add_format({'num_format': '###.00E+2',
                                             'font_color': 'black'})
        self.FNOK = self.WorkBook.add_format({'num_format': '###.00E+2',
                                              'font_color': 'red'})
        self.TmpPath = tempfile.mkdtemp(suffix='PyFET')
# Find devicelist that will be reported
        GroupBy = 'Devices.Name'
        self.DevicesList = Dban.FindCommonParametersValues(Table='DCcharacts',
                                                           Parameter=GroupBy,
                                                           Conditions=Conditions)
# Init Db connection
        self.Mydb = PyFETdb.PyFETdb(host='opter6.cnm.es',
                                    user='pyfet',
                                    passwd='p1-f3t17',
                                    db='pyFET')
#  Add Worksheets
        self.WorkBook.add_worksheet('Summary')
        for DevName in sorted(self.DevicesList):
            self.WorkBook.add_worksheet(DevName)
            if PathHistory is not None:
                Cond = {'Devices.Name=': (DevName, )}
                XlsHist = GenXlsDeviceHistory(PathHistory + DevName +'.xlsx', Cond)
                XlsHist.GenFullReport()
                XlsHist.close()

    def GenFullReport(self):
        for DevName in self.DevicesList:
            self.GenDeviceReport(DevName)
        self.GenSummaryReport()

    def GenDeviceReport(self, DeviceName):
        Sheet = self.WorkBook.sheetnames[DeviceName]
# Get Data
        Data, _ = self.GetDeviceData(DeviceName)
        self.FillDeviceInfoHeaders(Sheet)
        self.FillDeviceInfo(Sheet, DeviceName, Data)
# Insert data tables and Charts
        self.FillTrtValues(Sheet, Data, Loc=self.MeasInfoTableLoc)
        self.InsertCharFig(Sheet, Data)
        self.InsertCharMaps(Sheet, Data)
        plt.close('all')

    def GenSummaryReport(self):
        Sheet = self.WorkBook.sheetnames['Summary']
        self.FillDeviceInfoHeaders(Sheet, Vertical=False, Yeild=True)
        Sheet.set_column(0, 15, width=12)
        Yeild = len(self.ProbeMap)-1
# Fill device info fields and count Ok devices
        Grs = {}
        Counts = {}
        for k in self.DevOKFields.keys():
            Counts[k] = 0
        for idev, DevName in enumerate(sorted(self.DevicesList)):
            Data, Grs[DevName] = self.GetDeviceData(DevName)
            counts = self.FillDeviceInfo(Sheet, DevName, Data,
                                         LocOff=(idev, 0),
                                         Vertical=False,
                                         Yeild=Yeild)
            for k in self.DevOKFields.keys():
                Counts[k] += counts[k]
# Fill yeild
        if len(self.DevicesList) == 0:
            return
        row = idev + 2
        for k, val in self.DevOKFields.iteritems():
            col = val[1] + len(self.DevInfoFields)
            yeild = Counts[k]/float(Yeild*(idev+1))
            Sheet.write(row, col, yeild, self.FYeild)
        try:
# Insert Line plots
            for ipl, LiPlots in enumerate(self.SummaryLinePlots):
                Fig, _ = Dban.MultipleSearch(Groups=Grs, **LiPlots)
                fname = tempfile.mktemp(suffix='.png', dir=self.TmpPath)
                Fig.savefig(fname, dpi=self.FigsDpi)
                Sheet.insert_image(idev+4+ipl*self.SummaryPlotSpacing, 7, fname)
# Insert Boxplots
            for ipl, BoxPlots in enumerate(self.SummaryBoxPlots):
                Dban.MultipleSearchParam(Groups=Grs, **BoxPlots)
                fname = tempfile.mktemp(suffix='.png', dir=self.TmpPath)
                plt.gcf().savefig(fname, dpi=self.FigsDpi)
                Sheet.insert_image(idev+4+ipl*self.SummaryPlotSpacing, 0, fname)
        except:
            print 'Error plotting'

    def InsertCharFig(self, Sheet, Data):
        Plot = PyFETplt(Size=(12, 10))
        Plot.AddAxes(self.CharPars)
        Plot.PlotDataSet(DataDict=Data,
                         Trts=Data.keys())
        Plot.AddLegend(Axn=self.CharParLeg, fontsize='x-small')
        fname = tempfile.mktemp(suffix='.png', dir=self.TmpPath)
        Plot.Fig.savefig(fname, dpi=self.FigsDpi)

        if self.CharFigLoc[0] is None:
            Row = len(Data) + self.MeasInfoTableLoc[0] + 2
        else:
            Row = self.CharFigLoc[0]
        Sheet.insert_image(Row, self.CharFigLoc[1], fname)

    def InsertCharMaps(self, Sheet, Data):
        sfmt = ticker.ScalarFormatter(useMathText=True)
        sfmt.set_powerlimits((2, 2))
        for ip, Par in enumerate(self.MapPars):
            Map = CalcParMap(Data=Data,
                             ParMap=Par,
                             ParArgs=self.MapParArgs[ip],
                             ProbeMap=Cortical16Map)

            if Map is None:
                continue
            Fig, Ax = plt.subplots(figsize=self.MapsFigSize)
            Cax = Ax.imshow(Map, cmap=cm.afmhot, norm=self.MapNorm[ip])
            Ax.set_xlabel('column')
            Ax.set_ylabel('row')
            Ax.set_xticks(np.arange(self.ProbeMap['Shape'][0]))
            Ax.set_yticks(np.arange(self.ProbeMap['Shape'][1]))
            Ax.set_title(Par)

            cbar = Fig.colorbar(Cax, format=sfmt)
            cbar.set_label(self.MapUnits[ip], rotation=270, labelpad=10)

            fname = tempfile.mktemp(suffix='.png', dir=self.TmpPath)
            plt.tight_layout()
            Fig.savefig(fname, dpi=self.FigsDpi)
            col = self.MapLoc[1] + ip * self.MapSpacing
            Sheet.insert_image(self.MapLoc[0], col, fname)

    def CountOkDevices(self, DeviceName=None, Data=None,
                       Param=None, ParArgs=None,
                       RefVal=None, Lower=True):
        Count = 0
        if Data is None:
            Data, _ = self.GetDeviceData(DeviceName)
        for Trtn, Dat in sorted(Data.iteritems()):
            if not Dat[0].IsOK:
                continue

            if Param is 'IsOK' or Param is None:
                Count += 1
                continue

            func = Dat[0].__getattribute__('Get' + Param)
            val = func(**ParArgs)
            if val is None:
                continue
            if hasattr(val, '__iter__'):
                if val.size == 0:
                    continue

            if Lower:
                if val < RefVal:
                    Count += 1
            else:
                if val > RefVal:
                    Count += 1

        return Count

    def GetDeviceData(self, DeviceName):
        DeviceNames = (DeviceName, )

        CondBase = {}
        CondBase['Table'] = 'ACcharacts'
        CondBase['Last'] = True
        CondBase['GetGate'] = True
        CondBase['Conditions'] = {'Devices.Name=': DeviceNames,
                                  'CharTable.FuncStep=': ('Report', )}
        Data, _ = Dban.GetFromDB(**CondBase)
        if len(Data) > 0:
            print DeviceName, 'Getting data from Report Flag'
            return Data, CondBase

        CondBase['Conditions'] = {'Devices.Name=': DeviceNames}
        Data, _ = Dban.GetFromDB(**CondBase)
        if len(Data) > 0:
            print DeviceName, 'Getting data from last ACcharacts'
            return Data, CondBase

        CondBase['Table'] = 'DCcharacts'
        Data, _ = Dban.GetFromDB(**CondBase)
        if len(Data) > 0:
            print DeviceName, 'Getting data from last DCcharacts'
            return Data, CondBase

    def FillDeviceInfoHeaders(self, Sheet, LocOff=(0, 0),
                              Vertical=True, Yeild=None):
        ColOff = LocOff[1]
        RowOff = LocOff[0]
# Write headers of device info in rows
        for val in self.DevInfoFields.values():
            if Vertical:
                row = RowOff + val[1]
                col = ColOff
            else:
                row = RowOff
                col = ColOff + val[1]
            Sheet.write(row, col, val[0], self.Fbold)
# Write Headers for OK counts
        for val in self.DevOKFields.values():
            if Vertical:
                row = RowOff + val[1] + len(self.DevInfoFields)
                col = ColOff
            else:
                row = RowOff
                col = ColOff + val[1] + len(self.DevInfoFields)
            Sheet.write(row, col, val[0], self.Fbold)
# Write Headers for OK counts Yield
        if Yeild is not None:
            for val in self.DevOKFields.values():
                if Vertical:
                    row = RowOff + val[1] + len(self.DevInfoFields) + len(self.DevOKFields)
                    col = ColOff
                else:
                    row = RowOff
                    col = ColOff + val[1] + len(self.DevInfoFields) + len(self.DevOKFields)
                Sheet.write(row, col, 'Yeild ' + val[0], self.Fbold)

    def FillDeviceInfo(self, Sheet, DeviceName, Data,
                       LocOff=(0, 0), Vertical=True, Yeild=None):
        ColOff = LocOff[1]
        RowOff = LocOff[0]
# Fill Device info fields
        Tinf = self.Mydb.GetDevicesInfo(Conditions={'Devices.Name=': (DeviceName, )},
                                        Output=self.DevInfoFields.keys())
        for k, val in Tinf[0].iteritems():
            if Vertical:
                row = RowOff + self.DevInfoFields[k][1]
                col = ColOff + 1
            else:
                row = RowOff + 1
                col = ColOff + self.DevInfoFields[k][1]
            Sheet.write(row, col, val)
# Fill OK counts
        Counts = {}
        for k, v in self.DevOKFields.iteritems():
            if Vertical:
                row = RowOff + v[1] + len(self.DevInfoFields)
                col = ColOff + 1
            else:
                row = RowOff + 1
                col = ColOff + v[1] + len(self.DevInfoFields)
            count = self.CountOkDevices(Data=Data, Param=k, **v[2])
            Sheet.write(row, col, count)
            Counts[k] = count
# Fill OK counts Yield
        if Yeild is not None:
            for k, v in self.DevOKFields.iteritems():
                if Vertical:
                    row = RowOff + v[1] + len(self.DevInfoFields) + len(self.DevOKFields)
                    col = ColOff + 1
                else:
                    row = RowOff + 1
                    col = ColOff + v[1] + len(self.DevInfoFields) + len(self.DevOKFields)
                yeild = Counts[k]/float(Yeild)
                Sheet.write(row, col, yeild, self.FYeild)

        return Counts

    def FillTrtValues(self, Sheet, Data, Loc):
        RowOff = Loc[0]
        ColOff = Loc[1]
# Write Header
        Sheet.write(RowOff, ColOff, 'Trt Name', self.Fbold)
        Sheet.set_column(ColOff, ColOff, width=20)
        for Par, Parv in self.MeasFields.iteritems():
            Col = 1+ColOff+Parv[1]
            Sheet.write(RowOff, Col, Parv[0], self.Fbold)
            if Par == 'Time':
                Sheet.set_column(Col, Col, width=20)
        for val in self.TrtInfoFields.values():
            Col = 2 + ColOff + val[1] + len(self.MeasFields)
            Sheet.write(RowOff, Col, val[0], self.Fbold)
# Iter for each Trt in Data
        for iTrt, (Trtn, Dat) in enumerate(sorted(Data.iteritems())):
            Row = 1 + RowOff + iTrt
            Sheet.write(Row, 0, Trtn, self.Fbold)
            if Dat[0].IsOK:
                Format = self.FOK
            else:
                Format = self.FNOK
# Fill Meas fields
            for Par, Parv in self.MeasFields.iteritems():
                Col = 1 + ColOff + Parv[1]
                func = Dat[0].__getattribute__('Get' + Par)
                val = func(**Parv[2])
                if val is None:
                    continue
                if hasattr(val, '__iter__'):
                    if val.size == 0:
                        continue
                if Par == 'Time':
                    val = val[0, 0].astype(datetime.datetime).strftime('%x %X')
                    Sheet.write(Row, Col, val)
                    continue
                Sheet.write(Row, Col, val, Format)
# Fill Trt Info fields
            Tinf = self.Mydb.GetTrtsInfo(Conditions={'Trts.Name=': (Trtn, )},
                                         Output=self.TrtInfoFields.keys())
            for k, val in Tinf[0].iteritems():
                Col = 2 + ColOff + self.TrtInfoFields[k][1] + len(self.MeasFields)
                Sheet.write(Row, Col, val, Format)

    def close(self):
        self.WorkBook.close()
        shutil.rmtree(self.TmpPath)

