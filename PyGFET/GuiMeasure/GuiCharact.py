# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 16:03:58 2017

@author: user
"""
import os
import sys

from qtpy.QtWidgets import (QHeaderView, QCheckBox, QSpinBox, QLineEdit,
                            QDoubleSpinBox, QTextEdit, QComboBox,
                            QTableWidget, QAction, QMessageBox, QFileDialog,
                            QInputDialog)

from qtpy import QtWidgets, uic
#from qtpy.QtCore import Qt, QItemSelectionModel, QSettings

import matplotlib.pyplot as plt
import deepdish as dd
#import ctypes


#import PyGFET.DataStructures as PyData
import PyGFET.PlotDataClass as PyFETpl
import PyGFET.CharactCore as PyCharact
from PyGFET.RecordPlot import PltSlot, PlotRecord

#import PyDAQmx as Daq

#from ctypes import byref, c_int32
import numpy as np
#from scipy import signal
#import neo
import pickle
import quantities as pq
import inspect
import matplotlib.cm as cmx
import matplotlib.colors as mpcolors


class ContinuousAcquisitionPlots():

    def __init__(self, Rec):
        slots = []
        cmap = cmx.ScalarMappable(mpcolors.Normalize(vmin=0, vmax=len(Rec.SigNames.keys())),
                                  cmx.jet)

        for ind, sign in enumerate(sorted(Rec.SigNames.keys())):
            sl = PltSlot()
            sl.rec = Rec

            if sign.endswith('_DC'):
                sl.Position = 0
                sl.Color = cmap.to_rgba(ind)
                sl.DispName = sign

            if sign.endswith('_AC'):
                sl.Position = 1
                sl.Color = cmap.to_rgba(ind)
                sl.DispName = sign

            if sign.endswith('_Gate'):
                sl.Position = 2
                sl.DispName = sign

            if not sign.startswith('V'):
                sl.SigName = sign
                sl.OutType = 'V'
                slots.append(sl)

#            sl.Position = ind

#            if sign.endswith('_DC'):
#                sl.FiltType = ('lp', )
#                sl.FiltOrder = (2, )
#                sl.FiltF1 = (1, )

        #  Init Plot figures
        self.PltRecs = PlotRecord()
        self.PltRecs.CreateFig(slots, ShowLegend=True)
        plt.show()

    def PlotUpdate(self, Time):
        self.PltRecs.ClearAxes()
        self.PltRecs.PlotChannels(Time, Resamp=True)
        self.PltRecs
        self.PltRecs.Fig.canvas.draw()

    def __del__(self):
        plt.close('all')


class CharacLivePlot():

    DCPlotVars = ('Ids', 'Rds', 'Gm', 'Ig')
    BodePlotVars = ('GmPh', 'GmMag')
    PSDPlotVars = ('PSD',)
    PlotSwDC = None
    PlotSwAC = None
    DebugFig = None

    def __init__(self, SinglePoint=True, Bode=True, PSD=True, FFT=False):

        self.DebugFig, self.DebugAxs = plt.subplots()
        self.DebugAxs.ticklabel_format(axis='y', style='sci',
                                       scilimits=(-2, 2))
        plt.show()

        if not SinglePoint:
            self.PlotSwDC = PyFETpl.PyFETPlot()
            self.PlotSwDC.AddAxes(self.DCPlotVars)

        if Bode or PSD:
            PVAC = []
            if Bode:
                for var in self.BodePlotVars:
                    PVAC.append(var)
            if PSD:
                for var in self.PSDPlotVars:
                    PVAC.append(var)
            self.PlotSwAC = PyFETpl.PyFETPlot()
            self.PlotSwAC.AddAxes(PVAC)

        if FFT:
            self.FFTFig, self.FFTAxs = plt.subplots()
            self.FFTAxs.ticklabel_format(axis='y', style='sci',
                                         scilimits=(-2, 2))
            plt.show()

    def UpdateTimeViewPlot(self, Ids, Time, Dev):
        while self.DebugAxs.lines:
            self.DebugAxs.lines[0].remove()
        self.DebugAxs.plot(Time, Ids)
        self.DebugAxs.set_ylim(np.min(Ids), np.max(Ids))
        self.DebugAxs.set_xlim(np.min(Time), np.max(Time))
        self.DebugAxs.set_title(str(Dev))
        self.DebugFig.canvas.draw()

    def UpdateTimeAcViewPlot(self, Ids, Time):
        while self.DebugAxs.lines:
            self.DebugAxs.lines[0].remove()
        self.DebugAxs.plot(Time, Ids)
        self.DebugAxs.set_ylim(np.min(Ids), np.max(Ids))
        self.DebugAxs.set_xlim(np.min(Time), np.max(Time))
        self.DebugFig.canvas.draw()

    def UpdateSweepDcPlots(self, Dcdict):
        if self.PlotSwDC:
            self.PlotSwDC.ClearAxes()
            self.PlotSwDC.PlotDataCh(Dcdict)
            self.PlotSwDC.AddLegend()
            self.PlotSwDC.Fig.canvas.draw()

    def UpdateAcPlots(self, Acdict):
        if self.PlotSwAC:
            self.PlotSwAC.ClearAxes()
            self.PlotSwAC.PlotDataCh(Acdict)
#            self.PlotSwAC.AddLegend()
            self.PlotSwAC.Fig.canvas.draw()

    def PlotFFT(self, FFT):
        print 'CharacLivePlot PlotFFT'
        print FFT.shape
#        self.FFTFig, self.FFTAxs = plt.subplots()
#        self.FFTAxs.ticklabel_format(axis='y', style='sci',
#                                     scilimits=(-2, 2))
#        plt.show()

        self.FFTAxs.plot(np.abs(FFT))
#        self.FFTAxs.semilogx(FFT, OutFFT)
        self.FFTFig.canvas.draw()

    def __del__(self):
        plt.close('all')


###############################################################################
####
###############################################################################


#GuiTestDC_ui = "GuiTestDC_v3.ui"  # Enter file here.
#Ui_GuiTestDC, QtBaseClass = uic.loadUiType(GuiTestDC_ui)


class CharactAPP(QtWidgets.QMainWindow):
    OutFigFormats = ('svg', 'png')

    PlotCont = None
    PlotSweep = None
    FileName = None

    Charac = None  # intance of charact class

    def InitMenu(self):
        mainMenu = self.menubar
        fileMenu = mainMenu.addMenu('File')

        SaveFigAction = QAction('Save Figures', self)
        SaveFigAction.setShortcut('Ctrl+s')
        SaveFigAction.setStatusTip('Save all open figures')
        SaveFigAction.triggered.connect(self.SaveFigures)
        fileMenu.addAction(SaveFigAction)

        CloseFigsAction = QAction('Close Figures', self)
        CloseFigsAction.setStatusTip('Close all open figures')
        CloseFigsAction.triggered.connect(self.CloseFigures)
        fileMenu.addAction(CloseFigsAction)

        LoadConfAction = QAction('Load Configuration', self)
        LoadConfAction.setStatusTip('Load Config')
        LoadConfAction.triggered.connect(self.LoadConf)
        fileMenu.addAction(LoadConfAction)

        SaveConfAction = QAction('Save Configuration', self)
        SaveConfAction.setStatusTip('Save Config')
        SaveConfAction.triggered.connect(self.SaveConf)
        fileMenu.addAction(SaveConfAction)

    def __init__(self, parent=None):

        QtWidgets.QMainWindow.__init__(self)
        uipath = os.path.join(os.path.dirname(__file__), 'GuiCharact.ui')
        uic.loadUi(uipath, self)
        self.setWindowTitle('Characterization PyFET')

        self.InitMenu()

        # Buttons
        self.ButSweep.clicked.connect(self.ButSweepClick)
        self.ButInitChannels.clicked.connect(self.ButInitChannelsClick)
        self.ButUnselAll.clicked.connect(self.ButUnselAllClick)
        self.ButCont.clicked.connect(self.ButContClick)
        self.ButSaveCont.clicked.connect(self.SaveContData)

        # Combo Box
        self.CmbDevCond.currentIndexChanged.connect(self.DevCondChanged)

        # Check Box
        self.ChckSaveData.stateChanged.connect(self.ChckSaveDataChanged)

        # Slider
        self.SLTstart.valueChanged.connect(self.SLTStartChanged)
        self.SLTstop.valueChanged.connect(self.SLTStopChanged)

        # Spin Box
        self.SpnSVgsTP.valueChanged.connect(self.VgsTimePlotChanged)
        self.SpnSVdsTP.valueChanged.connect(self.VdsTimePlotChanged)

        # Signals Bode
        self.SpnFreqMin.valueChanged.connect(self.CheckBodeConfig)
        self.SpnFreqMax.valueChanged.connect(self.CheckBodeConfig)
        self.SpnNAvg.valueChanged.connect(self.CheckBodeConfig)

        # Signals PSD
        self.SpnPDSnFFT.valueChanged.connect(self.CheckPSDConfig)
        self.SpnAvg.valueChanged.connect(self.CheckPSDConfig)
        self.SpnFsPSD.valueChanged.connect(self.CheckPSDConfig)

        self.SweepEnableObjects = [self.SpnVgsMin,
                                   self.SpnVgsMax,
                                   self.SpnVgsStep,
                                   self.SpnVdsMin,
                                   self.SpnVdsMax,
                                   self.SpnVdsStep,
                                   self.SpnSVgs,
                                   self.SpnSVds,
                                   self.ChckSP,
                                   self.SpnFreqMin,
                                   self.ChckBode,
                                   self.ChckPSD,
                                   self.SpnFreqMax,
                                   self.SpnNFreqs,
                                   self.SpnAmp,
                                   self.SpnPDSnFFT,
                                   self.SpnAvg,
                                   self.SpnFsPSD,
                                   self.SpnInitCycle,
                                   self.SpnFinalCycle,
                                   self.ChckRhardware,
                                   self.ChckOutBode,
                                   self.SpnNAvg]

        self.ContEnableObjects = [self.SpnTestFreqMin,
                                  self.SpnTestFreqMax,
                                  self.SpnTestNFreqs,
                                  self.SpnTestAmp,
                                  self.SpnRefresh,
                                  self.SpnFsTime]

        self.ConfigTP = [self.chckTpDC,
                         self.chckTpAC]

# Init Channels
###############################################################################
    def ButUnselAllClick(self):
        for ck in self.GrChannels.findChildren(QtWidgets.QCheckBox):
            ck.setChecked(False)

    def ButInitChannelsClick(self):
        # Event InitChannels button
        Channels = self.GetSelectedChannels(self.GrChannels)
        GateChannels = self.GetSelectedChannels(self.GrChannelGate)
        self.GateCh = GateChannels
        if len(GateChannels) > 0:
            if len(GateChannels) > 1:
                QMessageBox.question(self, 'Message',
                                     "Warning: Select Only ONE Gate!",
                                     QMessageBox.Ok)
                return
            GateChannel = GateChannels[0]
        else:
            GateChannel = None

        if GateChannel in Channels:
            QMessageBox.question(self, 'Message', "Warning: Gate Duplicated!",
                                 QMessageBox.Ok)
            return

        if self.Charac is not None:
            self.Charac.__del__()

        Config = self.GetConfig(self.GrConfig)
        self.TimePlotConfig(Config)
        self.Charac = PyCharact.Charact(Channels=Channels,
                                        GateChannel=GateChannel,
                                        Configuration=Config)
#        self.Charac = Charact(Channels=Channels,
#                              GateChannel=GateChannel,
#                              Configuration=Config)

        # Define events callbacks
        self.Charac.EventCharSweepDone = self.CharSweepDoneCallBack
        self.Charac.EventCharBiasDone = self.CharBiasDoneCallBack
        self.Charac.EventCharBiasAttempt = self.CharBiasAttemptCallBack
        self.Charac.EventCharACDone = self.CharACDoneCallBack
        self.Charac.EventCharAcDataAcq = self.CharAcDataAcq

        self.Charac.EventContinuousDone = self.CharContDataCallback
        self.Charac.EventSetLabel = self.LabelsChanged
        self.Charac.EventSetBodeLabel = self.LabelsBodeChanged

        # Define Gains
        if Config in ('DC', 'AC'):
            self.Charac.IVGainAC = float(self.QGainDC.text())
        else:
            self.Charac.IVGainAC = float(self.QGainAC.text())
        self.Charac.IVGainDC = float(self.QGainDC.text())
        self.Charac.IVGainGate = float(self.QGainGate.text())
        self.Charac.Rhardware = float(self.QRhardware.text())

    def GetSelectedChannels(self, ChGroup):
        Chs = []
        for ck in ChGroup.findChildren(QtWidgets.QCheckBox):
            if ck.isChecked():
                Chs.append(str(ck.text()))
        return Chs  # Dictat amb els canals ['Ch08', 'Ch16', ...

    def GetConfig(self, ConfGroup):
        Config = []
        for n in ConfGroup.findChildren(QtWidgets.QCheckBox):
            if n.isChecked():
                Config.append(str(n.text()))
        return Config[0]

# Sweep
###############################################################################

    def ButSweepClick(self):
        if self.Charac is None:
            print 'Init Channels first'
            return

        # Event Start button
        if self.Charac.CharactRunning:
            print 'Stop'
            self.Charac.StopCharac()
        else:
            self.SetEnableObjects(val=False,
                                  Objects=self.SweepEnableObjects)
            if self.PlotSweep:
                del self.PlotSweep
            self.PlotSweep = CharacLivePlot(
                                        SinglePoint=self.ChckSP.isChecked(),
                                        Bode=self.ChckBode.isChecked(),
                                        PSD=self.ChckPSD.isChecked(),
                                        FFT=self.ChckFFT.isChecked())
            # Sweep Variables
            SwVgsVals, SwVdsVals = self.SweepVariables()

            # Init Cycles
            self.Cycle = self.SpnInitCycle.value()
            self.FinalCycle = self.SpnFinalCycle.value()

            # Set Charact configuration
            self.DevCondChanged()
            if self.ChckBode.isChecked():
                self.SetBodeConfig()
            if self.ChckPSD.isChecked():
                self.CheckPSDConfig()

            if self.ChckFFT.isChecked():
                print 'FFT checked'
                self.Charac.EventFFTDone = self.CharFFTCallBack

            if not self.ChckSaveData.isChecked():
                self.ChckSaveData.setChecked(True)

            self.Charac.InitSweep(VgsVals=SwVgsVals,
                                  VdsVals=SwVdsVals,
                                  PSD=self.ChckPSD.isChecked(),
                                  Bode=self.ChckBode.isChecked())

            if self.Charac.CharactRunning:
                self.ButSweep.setText('Stop')
            else:
                print 'ERROR'

    def SweepVariables(self):
        if self.ChckSP.isChecked():
            SwVgsVals = np.array([self.SpnSVgs.value(), ])
            SwVdsVals = np.array([self.SpnSVds.value(), ])
        else:
            SwVgsVals = np.linspace(self.SpnVgsMin.value(),
                                    self.SpnVgsMax.value(),
                                    self.SpnVgsStep.value())

            SwVdsVals = np.linspace(self.SpnVdsMin.value(),
                                    self.SpnVdsMax.value(),
                                    self.SpnVdsStep.value())
        return SwVgsVals, SwVdsVals

    def CheckBodeConfig(self):
        if self.Charac:
            self.SetBodeConfig()
            if self.Charac.BodeSignal:
                self.LblBodeDurationL.setText(str(
                        self.Charac.BodeSignal.BodeDuration[0]))
                self.LblBodeDurationH.setText(str(
                        self.Charac.BodeSignal.BodeDuration[1]))

    def LabelsBodeChanged(self, Vpp):
        if self.Charac:
            self.LblBodeDurationL.setText(str(
                    self.Charac.BodeSignal.BodeDuration[0]))
            self.LblBodeDurationH.setText(str(
                    self.Charac.BodeSignal.BodeDuration[1]))

            if self.Charac.BodeSignal.Vpp[1] is not None:
                self.LblVpp.setText('{:0.2e} - {:0.2e}'.format(
                        self.Charac.BodeSignal.Vpp[0],
                        self.Charac.BodeSignal.Vpp[1]))
            else:
                self.LblVpp.setText('{:0.2e}'.format(
                        self.Charac.BodeSignal.Vpp[0]))

    def SetBodeConfig(self):
        print 'Gui SetBodeConfig'
        if self.SpnFreqMin.value() and self.SpnFreqMax.value() > 0:
            self.Charac.SetBodeConfig(FreqMin=self.SpnFreqMin.value(),
                                      FreqMax=self.SpnFreqMax.value(),
                                      nFreqs=self.SpnNFreqs.value(),
                                      Arms=self.SpnAmp.value(),
                                      nAvg=self.SpnNAvg.value(),
                                      BodeRh=self.ChckRhardware.isChecked(),
                                      BodeOut=self.ChckOutBode.isChecked(),
                                      RemoveDC=self.ChckRemoveDC.isChecked())

    def CheckPSDConfig(self):
        if self.Charac:
            self.SetPSDConfig()
            self.Charac.PSDDuration = self.Charac.PSDnFFT*self.Charac.PSDnAvg*(
                    1/self.Charac.PSDFs)
            self.LblPSDDuration.setText(str(self.Charac.PSDDuration))

    def SetPSDConfig(self):
        self.Charac.PSDnFFT = 2**self.SpnPDSnFFT.value()
        self.Charac.PSDFs = self.SpnFsPSD.value()
        self.Charac.PSDnAvg = self.SpnAvg.value()

    def SetEnableObjects(self, val, Objects):
        for obj in Objects:
            obj.setEnabled(val)

    def DevCondChanged(self):
        if self.Charac:
            self.Charac.DevCondition = float(self.CmbDevCond.currentText())

    def LabelsChanged(self, Vds, Vgs):
        self.LblVds.setText(str(Vds))
        self.LblVgs.setText(str(Vgs))

    def NextCycle(self):
        if self.Cycle < self.FinalCycle-1:
            self.Cycle += 1
            self.LblCycle.setText(str(self.Cycle))

            SwVgsVals, SwVdsVals = self.SweepVariables()
            self.Charac.InitSweep(VgsVals=SwVgsVals,
                                  VdsVals=SwVdsVals,
                                  PSD=self.ChckPSD.isChecked(),
                                  Bode=self.ChckBode.isChecked())
        else:
            if self.Charac.CharactRunning:
                self.Cycle = 0
                self.LblCycle.setText(str(self.Cycle))
                self.Charac.SetBias(Vds=0, Vgs=0)
                self.StopSweep()
                self.Charac.StopCharac()

# Continuous Acquisition
###############################################################################
    def ButContClick(self):  # Evento button TimeCont
        if self.Charac is None:
            print 'Init Channels first'
            return

        if self.Charac.CharactRunning:
            self.Charac.StopCharac()
            self.ButCont.setText('Start')
            self.SetEnableObjects(val=True, Objects=self.ContEnableObjects)
            self.SaveContData()
            self.Charac.ContRecord = None

        else:
            self.SetEnableObjects(val=False,
                                  Objects=self.ContEnableObjects)
            self.ButCont.setText('Stop')
            if self.PlotCont:
                del self.PlotCont
            if not self.chckTpAC.isChecked() and not self.chckTpDC.isChecked():
                QMessageBox.question(self, 'Message',
                                     "Warning: Select One Acquisition type!",
                                     QMessageBox.Ok)
                self.ButCont.setText('Start')
                return
            self.SetTestSignalConfig()
            self.Charac.InitContMeas(Vds=self.SpnSVdsTP.value(),
                                     Vgs=self.SpnSVgsTP.value(),
                                     Fs=self.SpnFsTime.value(),
                                     Refresh=self.SpnRefresh.value(),
                                     RecDC=self.chckTpDC.isChecked(),
                                     RecAC=self.chckTpAC.isChecked(),
                                     RecGate=self.GateCh,
                                     GenTestSig=self.chckTestSig.isChecked())
            self.PlotCont = ContinuousAcquisitionPlots(self.Charac.ContRecord)

            if self.Charac.CharactRunning:
                self.ButCont.setText('Stop')
            else:
                print 'ERROR'

    def TimePlotConfig(self, Config):
        if Config == 'DC':
            self.chckTpDC.setChecked(True)
            self.chckTpAC.setChecked(False)
        elif Config == 'AC':
            self.chckTpDC.setChecked(False)
            self.chckTpAC.setChecked(True)
        else:
            self.chckTpDC.setChecked(True)
            self.chckTpAC.setChecked(True)

        self.SetEnableObjects(val=False, Objects=self.ConfigTP)

    def SetTestSignalConfig(self):
        print 'Gui SetTestSignalConfig'
        self.Charac.SetContSig(FreqMin=self.SpnTestFreqMin.value(),
                               FreqMax=self.SpnTestFreqMax.value(),
                               nFreqs=self.SpnTestNFreqs.value(),
                               Arms=self.SpnTestAmp.value())

    def VgsTimePlotChanged(self):
        if self.Charac.CharactRunning:
            self.Charac.SetBias(Vds=self.SpnSVdsTP.value(),
                                Vgs=self.SpnSVgsTP.value())

    def VdsTimePlotChanged(self):
        if self.Charac.CharactRunning:
            self.Charac.SetBias(Vds=self.SpnSVdsTP.value(),
                                Vgs=self.SpnSVgsTP.value())

    def SLTStartChanged(self):
        if self.SLTstop.value() <= self.SLTstart.value():
            self.SLTstop.setValue(self.SLTstart.value()+self.SpnWindow.value())
        self.SLTStopChanged()

    def SLTStopChanged(self):
        if self.SLTstop.value() <= self.SLTstart.value():
            if self.SLTstop.value() == 0:
                self.SLTstop.setValue(1)
                self.SLTstart.setValue(self.SLTstop.value()-1)
            self.SLTstart.setValue(self.SLTstop.value()-1)
        time = (self.SLTstart.value()*pq.s, self.SLTstop.value()*pq.s)

        if self.ChckPauseCont.isChecked():
            if self.chckTpAC.isChecked():
                Name = self.Charac.ChNamesList[0] + '_AC'
            else:
                Name = self.Charac.ChNamesList[0] + '_DC'
            tstop = self.Charac.ContRecord.Signal(ChName=Name).t_stop
            self.SLTstart.setMaximum(tstop)
            self.SLTstop.setMaximum(tstop)
            self.LblTstartMax.setText(str(tstop))
            self.LblStopMax.setText(str(tstop))

            self.PlotCont.PltRecs.ClearAxes()
            time = (self.SLTstart.value()*pq.s, self.SLTstop.value()*pq.s)
            self.PlotCont.PlotUpdate(Time=time)

# Events Done
###############################################################################

    def CharSweepDoneCallBack(self, Dcdict, Acdict):
        print 'Gui sweep done save data'
        print Dcdict, Acdict
        if self.ChckSaveData.isChecked():
            Filename = self.FileName + "{}-Cy{}.h5".format('', self.Cycle)
            self.LblPath.setText(Filename)
            if Acdict:
                dd.io.save(Filename, (Dcdict, Acdict), ('zlib', 1))
#                pickle.dump(Acdict, open('SaveDcData.pkl', 'wb'))
            else:
                dd.io.save(Filename, Dcdict, ('zlib', 1))
        self.NextCycle()

    def CharBiasDoneCallBack(self, Dcdict):
        print 'Gui bias done refresh'
        self.PlotSweep.UpdateSweepDcPlots(Dcdict)

    def CharFFTCallBack(self, FFT):
        print 'Gui FFT done callback'
        print FFT.shape
        if self.ChckFFT.isChecked():
            self.PlotSweep.PlotFFT(FFT[1:])

    def CharAcDataAcq(self, Ids, time):
        print 'Gui AC attemp refresh'
        self.PlotSweep.UpdateTimeAcViewPlot(Ids, time)

    def CharACDoneCallBack(self, Acdict):
        print' Gui ACPlots done refresh'
        self.PlotSweep.UpdateAcPlots(Acdict)
        if not self.Charac.CharactRunning:
            self.StopSweep()

    def CharBiasAttemptCallBack(self, Ids, Time, Dev):
        print 'Gui attemp refresh'
        self.PlotSweep.UpdateTimeViewPlot(Ids, Time, Dev)
        if not self.Charac.CharactRunning:
            self.StopSweep()

    def CharContDataCallback(self, tstop):
        print 'Gui Continuous Data Done Callback'
        if not self.ChckPauseCont.isChecked():
            time = (tstop - self.SpnWindow.value()*pq.s, tstop)
            print tstop, time
            if self.PlotCont:
                self.PlotCont.PlotUpdate(Time=time)

# Stop Events
###############################################################################
    def StopSweep(self):
        print 'Stop'
        self.SetEnableObjects(val=True, Objects=self.SweepEnableObjects)
        self.Charac.SetBias(Vds=0, Vgs=0)
        self.ButSweep.setText('Start')
        self.ChckSaveData.setChecked(False)

# Save Data Events
###############################################################################
    def SaveContData(self):
        name, _ = QFileDialog.getSaveFileName(self, 'Save File')
        if not name:
            return
        else:
            self.Charac.ContRecord.SaveRecord(name + '.h5')

    def ChckSaveDataChanged(self):
        if self.ChckSaveData.isChecked():
            self.FileName, _ = QFileDialog.getSaveFileName(self, 'Save File')
            print self.FileName
            if not self.FileName:
                self.ChckSaveData.setChecked(False)
                return
            self.LblPath.setText(self.FileName)
        else:
            self.FileName = None
            self.LblPath.setText('')

# Configuration & Figures Menu
###############################################################################
    def SaveConf(self):
        fileName, _ = QFileDialog.getSaveFileName(self, "Export Data", "",
                                                  "Pickle Files (*.pkl);; All Files (*)")
        if not fileName:
            return
        self.guisave(fileName)

    def LoadConf(self):
        LoadFileName = QFileDialog.getOpenFileName(self)
        if LoadFileName[0]:
            self.guirestore(LoadFileName[0])

    def guisave(self, FileName):

        Configuration = {}
        for name, obj in inspect.getmembers(self):
            if isinstance(obj, QCheckBox):
                Configuration[obj.objectName()] = obj.checkState()
            elif isinstance(obj, QSpinBox) or isinstance(obj, QDoubleSpinBox):
                Configuration[obj.objectName()] = obj.value()
            elif isinstance(obj, QLineEdit):
                Configuration[obj.objectName()] = obj.text()
            elif isinstance(obj, QComboBox):
                Configuration[obj.objectName()] = obj.currentIndex()
        pickle.dump(Configuration, open(FileName, 'wb'))

    def guirestore(self, LoadFileName):

        Configuration = pickle.load(open(LoadFileName))
        for nom, obj in inspect.getmembers(self):

            if isinstance(obj, QCheckBox):
                if obj.objectName() in Configuration:
                    obj.setCheckState(Configuration[obj.objectName()])

            elif isinstance(obj, QSpinBox) or isinstance(obj, QDoubleSpinBox):
                if obj.objectName() in Configuration:
                    obj.setValue(Configuration[obj.objectName()])

            elif isinstance(obj, QLineEdit):
                if obj.objectName() in Configuration:
                    obj.setText(Configuration[obj.objectName()])

            elif isinstance(obj, QComboBox):
                if obj.objectName() in Configuration:
                    obj.setCurrentIndex(Configuration[obj.objectName()])

    def CloseFigures(self):
        plt.close('all')

    def SaveFigures(self):
        Dir = QFileDialog.getExistingDirectory(self)
        Prefix, okPressed = QInputDialog.getText(self,
                                                 'Prefix',
                                                 'Prefix for files',
                                                 text='Figure')
        if Dir and okPressed:
            for i in plt.get_fignums():
                plt.figure(i)
                for ext in self.OutFigFormats:
                    fileOut = Dir + '/' + Prefix + '{}.' + ext
                    plt.savefig(fileOut.format(i))


def main():
    import argparse
    import pkg_resources

    # Add version option
    __version__ = pkg_resources.require("PyGFET")[0].version
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', action='version',
                        version='%(prog)s {version}'.format(
                            version=__version__))
    parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    w = CharactAPP()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
