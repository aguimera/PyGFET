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
from qtpy.QtCore import Qt, QItemSelectionModel, QSettings

import matplotlib.pyplot as plt
import deepdish as dd
import ctypes

import PyGFET.DataStructures as PyData
import PyGFET.PlotDataClass as PyFETpl
from PyGFET.PyFETRecord import NeoRecord, PltSlot, PlotRecord

import PyDAQmx as Daq

from ctypes import byref, c_int32
import numpy as np
from scipy import signal
import neo
import pickle
import quantities as pq
import inspect
import matplotlib.cm as cmx
import matplotlib.colors as mpcolors

"""This example is a PyDAQmx version of the ContAcq_IntClk.c example
It illustrates the use of callback functions

This example demonstrates how to acquire a continuous amount of
data using the DAQ device's internal clock. It incrementally stores the data
in a Python list.
"""

###############################################################################
######
###############################################################################


class ReadAnalog(Daq.Task):

    '''
    Class to read data from Daq card

    TODO - Implement the callback option to read data
    '''

#    Events list
    EveryNEvent = None
    DoneEvent = None

    ContSamps = False
    EverySamps = 1000

    def __init__(self, InChans, Range=5.0):

        Daq.Task.__init__(self)
        self.Channels = InChans

        Dev = self.GetDevName()
        for Ch in self.Channels:
            self.CreateAIVoltageChan(Dev.format(Ch), "",
                                     Daq.DAQmx_Val_RSE,
                                     -Range, Range,
                                     Daq.DAQmx_Val_Volts, None)

        self.AutoRegisterDoneEvent(0)

    def GetDevName(self,):
        # Get Device Name of Daq Card
        n = 1024
        buff = ctypes.create_string_buffer(n)
        Daq.DAQmxGetSysDevNames(buff, n)
        if sys.version_info >= (3,):
            value = buff.value.decode()
        else:
            value = buff.value
        Dev = value + '/{}'

        return Dev

    def ReadData(self, Fs=1000, nSamps=10000, EverySamps=1000):
        print 'ReadAnalog ReadData'
        print 'Fs', Fs
        print 'nSamps', nSamps
        print 'EverySamps', EverySamps

        self.Fs = Fs
        self.EverySamps = EverySamps

        self.data = np.ndarray([len(self.Channels), ])

        self.CfgSampClkTiming("", Fs, Daq.DAQmx_Val_Rising,
                              Daq.DAQmx_Val_FiniteSamps, nSamps)

        self.AutoRegisterEveryNSamplesEvent(Daq.DAQmx_Val_Acquired_Into_Buffer,
                                            self.EverySamps, 0)
        self.StartTask()

    def ReadContData(self, Fs=1000, EverySamps=1000):
        print 'ReadAnalog ReadContData'
        print 'Fs', Fs
        print 'EverySamps', EverySamps

        self.Fs = Fs
        self.EverySamps = np.int32(EverySamps)
        self.ContSamps = True  # TODO check it

        self.CfgSampClkTiming("", Fs, Daq.DAQmx_Val_Rising,
                              Daq.DAQmx_Val_ContSamps, self.EverySamps*100)

        self.AutoRegisterEveryNSamplesEvent(Daq.DAQmx_Val_Acquired_Into_Buffer,
                                            self.EverySamps, 0)

        self.StartTask()

    def StopContData(self):
        self.StopTask()
        self.ContSamps = False

    def EveryNCallback(self):
        print 'ReadAnalog every N'
        read = c_int32()
        data = np.zeros((self.EverySamps, len(self.Channels)))
        self.ReadAnalogF64(self.EverySamps, 10.0,
                           Daq.DAQmx_Val_GroupByScanNumber,
                           data, data.size, byref(read), None)

        if not self.ContSamps: # TODO check why stack here
            self.data = np.vstack((self.data, data))

#        print data.size, self.data.shape

        if self.EveryNEvent:
            self.EveryNEvent(data)

    def DoneCallback(self, status):
        print 'ReadAnalog Done'
        self.StopTask()
        self.UnregisterEveryNSamplesEvent()

        if self.DoneEvent:
            self.DoneEvent(self.data[1:, :])# TODO check why 1:

        return 0  # The function should return an integer

###############################################################################
#####
###############################################################################


class WriteAnalog(Daq.Task):

    '''
    Class to write data to Daq card
    '''
    def __init__(self, Channels):

        Daq.Task.__init__(self)
        Dev = self.GetDevName()        
        for Ch in Channels:
            self.CreateAOVoltageChan(Dev.format(Ch), "",
                                     -5.0, 5.0, Daq.DAQmx_Val_Volts, None)
        self.DisableStartTrig()
        self.StopTask()

    def GetDevName(self,):
        # Get Device Name of Daq Card
        n = 1024
        buff = ctypes.create_string_buffer(n)
        Daq.DAQmxGetSysDevNames(buff, n)
        if sys.version_info >= (3,):
            value = buff.value.decode()
        else:
            value = buff.value
        Dev = value + '/{}'

        return Dev

    def SetVal(self, value):

        self.StartTask()
        self.WriteAnalogScalarF64(1, -1, value, None)
        self.StopTask()

    def SetSignal(self, Signal, nSamps):

        read = c_int32()

        self.CfgSampClkTiming('ai/SampleClock', 1, Daq.DAQmx_Val_Rising,
                              Daq.DAQmx_Val_FiniteSamps, nSamps)

        self.CfgDigEdgeStartTrig('ai/StartTrigger', Daq.DAQmx_Val_Rising)
        self.WriteAnalogF64(nSamps, False, -1, Daq.DAQmx_Val_GroupByChannel,
                            Signal, byref(read), None)
        self.StartTask()

    def SetContSignal(self, Signal, nSamps):
        read = c_int32()

        self.CfgSampClkTiming('ai/SampleClock', 1, Daq.DAQmx_Val_Rising,
                              Daq.DAQmx_Val_ContSamps, nSamps)

        self.CfgDigEdgeStartTrig('ai/StartTrigger', Daq.DAQmx_Val_Rising)
        self.WriteAnalogF64(nSamps, False, -1, Daq.DAQmx_Val_GroupByChannel,
                            Signal, byref(read), None)
        self.StartTask()

###############################################################################
#####
###############################################################################


class ChannelsConfig():

    # Daq card connections mapping 'Chname' : (DCout, ACout)
    aiChannels = {'Ch01': ('ai0', 'ai8'),
                  'Ch02': ('ai1', 'ai9'),
                  'Ch03': ('ai2', 'ai10'),
                  'Ch04': ('ai3', 'ai11'),
                  'Ch05': ('ai4', 'ai12'),
                  'Ch06': ('ai5', 'ai13'),
                  'Ch07': ('ai6', 'ai14'),
                  'Ch08': ('ai7', 'ai15'),
                  'Ch09': ('ai16', 'ai24'),
                  'Ch10': ('ai17', 'ai25'),
                  'Ch11': ('ai18', 'ai26'),
                  'Ch12': ('ai19', 'ai27'),
                  'Ch13': ('ai20', 'ai28'),
                  'Ch14': ('ai21', 'ai29'),
                  'Ch15': ('ai22', 'ai30'),
                  'Ch16': ('ai23', 'ai31')}

# ChannelIndex = {'Ch01': (0-31, 0-15)}-->> {Chname: (input index, sort index)}
    DCChannelIndex = None
    ACChannelIndex = None
    GateChannelIndex = None

    ChNamesList = None
    # ReadAnalog class with all channels
    Inputs = None
    InitConfig = None

    # Events list
    DCDataDoneEvent = None
    DCDataEveryNEvent = None
    ACDataDoneEvent = None
    ACDataEveryNEvent = None
    GateDataDoneEvent = None
    GateDataEveryNEvent = None

    def DelInputs(self):
        self.Inputs.ClearTask()

    def InitInputs(self, Channels, GateChannel=None, Configuration='Both'):
        if self.Inputs is not None:
            self.DelInputs()

        InChans = []

        self.ChNamesList = sorted(Channels)
        self.DCChannelIndex = {}
        self.ACChannelIndex = {}
        index = 0
        sortindex = 0
        for ch in sorted(Channels):
            if Configuration in ('DC', 'Both'):
                InChans.append(self.aiChannels[ch][0])
                self.DCChannelIndex[ch] = (index, sortindex)
                index += 1
            if Configuration in ('AC', 'Both'):
                InChans.append(self.aiChannels[ch][1])
                self.ACChannelIndex[ch] = (index, sortindex)
                index += 1
            sortindex += 1

        if GateChannel:
            self.GateChannelIndex = {GateChannel: (index, 0)}
            InChans.append(self.aiChannels[GateChannel][0])
        else:
            self.GateChannelIndex = None

        print 'Channels configurtation'
        print 'Gate', self.GateChannelIndex
        print 'Channels ', len(self.ChNamesList)
        print 'ai list ->', InChans
        for ch in sorted(Channels):
            if Configuration == 'DC':
                print ch, ' DC -> ', self.aiChannels[ch][0], self.DCChannelIndex[ch]
                self.ACChannelIndex = self.DCChannelIndex
            elif Configuration == 'AC':
                print ch, ' AC -> ', self.aiChannels[ch][1], self.ACChannelIndex[ch]
                self.DCChannelIndex = self.ACChannelIndex
            else:
                print ch, ' DC -> ', self.aiChannels[ch][0], self.DCChannelIndex[ch]
                print ch, ' AC -> ', self.aiChannels[ch][1], self.ACChannelIndex[ch]

        self.Inputs = ReadAnalog(InChans=InChans)

        # events linking
        self.Inputs.EveryNEvent = self.EveryNEventCallBack
        self.Inputs.DoneEvent = self.DoneEventCallBack

    def __init__(self, Channels, GateChannel=None, Configuration='Both',
                 ChVg='ao2', ChVs='ao1', ChVds='ao0', ChVsig='ao3'):

        self.InitConfig = {}
        self.InitConfig['Channels'] = Channels
        self.InitConfig['GateChannel'] = GateChannel
        self.InitConfig['Configuration'] = Configuration

        self.InitInputs(Channels=Channels,
                        GateChannel=GateChannel,
                        Configuration=Configuration)

        # Output Channels
        self.VsOut = WriteAnalog((ChVs,))
        self.VdsOut = WriteAnalog((ChVds,))
        self.VgOut = WriteAnalog((ChVg,))
        self.Vsig = WriteAnalog((ChVsig,))

    def SetBias(self, Vds, Vgs):
        print 'ChannelsConfig SetBias Vgs ->', Vgs, 'Vds ->', Vds
        self.VdsOut.SetVal(Vds)
        self.VsOut.SetVal(-Vgs)
        self.BiasVd = Vds-Vgs

    def SetSignal(self, Signal, nSamps):
        if not self.VgOut:
            self.VgOut = WriteAnalog(('ao2',))
        self.VgOut.DisableStartTrig()
        self.VgOut.SetSignal(Signal=Signal,
                             nSamps=nSamps)

    def SetContSignal(self, Signal, nSamps):
        if not self.VgOut:
            self.VgOut = WriteAnalog(('ao2',))
        self.VgOut.DisableStartTrig()
        self.VgOut.SetContSignal(Signal=Signal,
                                 nSamps=nSamps)

    def _SortChannels(self, data, SortDict):
        (samps, inch) = data.shape
        sData = np.zeros((samps, len(SortDict)))
        print samps, inch, data.shape, sData.shape
        for chn, inds in SortDict.iteritems():
            sData[:, inds[1]] = data[:, inds[0]]
        return sData

    def EveryNEventCallBack(self, Data):
        _DCDataEveryNEvent = self.DCDataEveryNEvent
        _GateDataEveryNEvent = self.GateDataEveryNEvent
        _ACDataEveryNEvent = self.ACDataEveryNEvent

        if _GateDataEveryNEvent:
            _GateDataEveryNEvent(self._SortChannels(Data,
                                                    self.GateChannelIndex))
        if _DCDataEveryNEvent:
            _DCDataEveryNEvent(self._SortChannels(Data,
                                                  self.DCChannelIndex))
        if _ACDataEveryNEvent:
            _ACDataEveryNEvent(self._SortChannels(Data,
                                                  self.ACChannelIndex))

    def DoneEventCallBack(self, Data):
        if self.VgOut:
            self.VgOut.StopTask()

        _DCDataDoneEvent = self.DCDataDoneEvent
        _GateDataDoneEvent = self.GateDataDoneEvent
        _ACDataDoneEvent = self.ACDataDoneEvent

        if _GateDataDoneEvent:
            _GateDataDoneEvent(self._SortChannels(Data,
                                                  self.GateChannelIndex))
        if _DCDataDoneEvent:
            _DCDataDoneEvent(self._SortChannels(Data,
                                                self.DCChannelIndex))
        if _ACDataDoneEvent:
            _ACDataDoneEvent(self._SortChannels(Data,
                                                self.ACChannelIndex))

    def ReadChannelsData(self, Fs=1000, nSamps=10000, EverySamps=1000):
        self.Inputs.ReadData(Fs=Fs,
                             nSamps=nSamps,
                             EverySamps=EverySamps)

    def __del__(self):
        print 'Delete class'
        if self.VgOut:
            self.VgOut.ClearTask()
        self.VdsOut.ClearTask()
        self.VsOut.ClearTask()
        self.Inputs.ClearTask()
#        del(self.VgOut)
#        del(self.VdsOut)
#        del(self.VsOut)
#        del(self.Inputs)

###############################################################################
#####
###############################################################################


class FFTConfig():
    Freqs = None
    Fs = None
    nFFT = None
    Finds = None
    AcqSequential = False


class FFTTestSignal():

    FsH = 2e6
    FsL = 1000
    FMinLow = 0.5  # Lowest freq to acquire in 1 time
    FThres = 10  # For two times adq split freq

    FFTconfs = [FFTConfig(), ]

    Fsweep = None
    FFTAmps = None

    BodeDuration = [None, None]
    Vpp = [None, None]
    AddnFFT = 2
    nAvg = 2
    Arms = 1e-3
    EventFFTDebug = None

    def __init__(self, FreqMin, FreqMax, nFreqs, Arms, nAvg):

        self.nAvg = nAvg
        self.Arms = Arms

        if FreqMax > self.FsH/2:
            FreqMax = self.FsH/2 - 1

        if FreqMax <= self.FThres:
            FreqMax = self.FThres + 1

        fsweep = np.logspace(np.log10(FreqMin), np.log10(FreqMax), nFreqs)
        if np.any(fsweep < self.FMinLow) and nFreqs > 1:  # TODO Check this
            self.FFTconfs.append(FFTConfig())

            self.FFTconfs[0].nFFT = int(2**((np.around(np.log2(self.FsH/self.FThres))+1)+self.AddnFFT))
            self.FFTconfs[1].nFFT = int(2**((np.around(np.log2(self.FsL/FreqMin))+1)+self.AddnFFT))
            self.FFTconfs[0].Fs = self.FsH
            self.FFTconfs[1].Fs = self.FsL

            # LowFrequencies
            FreqsL, indsL = self.CalcCoherentSweepFreqs(FreqMin=FreqMin,
                                                        FreqMax=np.max(fsweep[np.where(self.FThres>fsweep)]),
                                                        nFreqs=np.sum(fsweep < self.FThres),
                                                        Fs=self.FFTconfs[1].Fs,
                                                        nFFT=self.FFTconfs[1].nFFT)

            # HighFrequencies
            FreqsH, indsH = self.CalcCoherentSweepFreqs(FreqMin=np.min(fsweep[np.where(fsweep>self.FThres)]),
                                                        FreqMax=FreqMax,
                                                        nFreqs=np.sum(self.FThres < fsweep),
                                                        Fs=self.FFTconfs[0].Fs,
                                                        nFFT=self.FFTconfs[0].nFFT)
            self.FFTconfs[0].Finds = indsH
            self.FFTconfs[0].Freqs = FreqsH

            self.FFTconfs[1].Finds = indsL
            self.FFTconfs[1].Freqs = FreqsL

            self.Fsweep = np.hstack((self.FFTconfs[1].Freqs,
                                     self.FFTconfs[0].Freqs))

            self.FFTconfs[0].AcqSequential = True

            self.BodeDuration = [self.FFTconfs[0].nFFT*(1/float(self.FFTconfs[0].Fs))*self.nAvg,
                                 self.FFTconfs[1].nFFT*(1/float(self.FFTconfs[1].Fs))*self.nAvg]

        else:
            if len(self.FFTconfs) > 1:
                del(self.FFTconfs[1])
            self.FFTconfs[0].nFFT = int(2**((np.around(np.log2(self.FsH/FreqMin))+1)+self.AddnFFT))
            self.FFTconfs[0].Fs = self.FsH

            FreqsH, indsH = self.CalcCoherentSweepFreqs(FreqMin=FreqMin,
                                                        FreqMax=FreqMax,
                                                        nFreqs=nFreqs,
                                                        Fs=self.FFTconfs[0].Fs,
                                                        nFFT=self.FFTconfs[0].nFFT)
            self.FFTconfs[0].Finds = indsH
            self.FFTconfs[0].Freqs = FreqsH

            self.Fsweep = self.FFTconfs[0].Freqs
            self.FFTconfs[0].AcqSequential = True
            self.BodeDuration = [self.FFTconfs[0].nFFT*(1/float(self.FFTconfs[0].Fs))*self.nAvg, None]

    def CalcCoherentSweepFreqs(self, FreqMin, FreqMax, nFreqs, Fs, nFFT):
        nmin = (FreqMin*(nFFT))/Fs
        nmax = (FreqMax*(nFFT))/Fs
        freqs = np.round(
                np.logspace(np.log10(nmin), np.log10(nmax), nFreqs), 0)
        Freqs = (float(Fs)/(nFFT))*np.unique(freqs)

        # Calc Indexes
        freqs = np.fft.rfftfreq(nFFT, 1/float(Fs))
#        freqs = [np.round(x, 7) for x in freq]
#        Freqs = [np.round(x, 7) for x in Freqs]
#        freqs = np.array(freqs)
#        Freqs = np.array(Freqs)
        Inds = np.where(np.in1d(freqs, Freqs))[0]

        return Freqs, Inds

    def GenSignal(self, Ind=0, Delay=0):

        FFTconf = self.FFTconfs[Ind]

        Ts = 1/float(FFTconf.Fs)
        Ps = FFTconf.nFFT * self.nAvg * Ts

        Amp = self.Arms*np.sqrt(2)
        t = np.arange(0, Ps, Ts) + Delay
        out = np.zeros(t.size)
        for f in np.nditer(FFTconf.Freqs):
            s = Amp*np.sin(f*2*np.pi*(t))
            out = out + s

        self.FFTAmps = (2*np.fft.rfft(
                out, FFTconf.nFFT)/FFTconf.nFFT)[FFTconf.Finds]
        self.Vpp[Ind] = np.max(out) + np.abs(np.min(out))
#        self.FFTconfs[Ind].Vpp = np.max(out) + np.abs(np.min(out))

        out[-1] = 0

        return out, t

    def CalcFFT(self, Data, Ind):
        FFTconf = self.FFTconfs[Ind]

        a = Data.reshape((self.nAvg, FFTconf.nFFT))
        acc = np.zeros(((FFTconf.nFFT/2)+1))
        for w in a:
            acc = acc + (2 * np.fft.rfft(w, FFTconf.nFFT) / FFTconf.nFFT)

        Out = (acc/self.nAvg)[FFTconf.Finds]

        if self.EventFFTDebug is not None:
            self.EventFFTDebug(acc/self.nAvg)

        return Out

###############################################################################
#####
###############################################################################


class FFTBodeAnalysis():

    BodeRh = None
    BodeRhardware = 150e3
    BodeSignal = None
    RemoveDC = None

#    AdqDelay = 0
    AdqDelay = -5.8e-7

    def SetBodeConfig(self, FreqMin=0.1, FreqMax=15000, nFreqs=10, Arms=10e-3,
                      nAvg=1, BodeRh=False, BodeOut=False, RemoveDC=False):
        print 'FFTBodeAnalysis SetBodeConfig'

        self.BodeSignal = FFTTestSignal(FreqMin=FreqMin,
                                        FreqMax=FreqMax,
                                        nFreqs=nFreqs,
                                        Arms=Arms,
                                        nAvg=nAvg)

        self.BodeRh = BodeRh
        self.BodeOut = BodeOut
        self.RemoveDC = RemoveDC

    def SetContSig(self, FreqMin=0.1, FreqMax=15000, nFreqs=10, Arms=10e-3):
        print 'FFTBodeAnalysis SetContSig'
        self.ContSig = FFTTestSignal(FreqMin=FreqMin,
                                     FreqMax=FreqMax,
                                     nFreqs=nFreqs,
                                     Arms=Arms,
                                     nAvg=1)
        self.ContSig.FMinLow = None
        self.ContSig.FsH = 1000
        self.ContSig.__init__(FreqMin=FreqMin,
                              FreqMax=FreqMax,
                              nFreqs=nFreqs,
                              Arms=Arms,
                              nAvg=1)

    def CalcBode(self, Data, Ind, IVGainAC, ChIndexes):
        print 'CalcBode'
        if self.BodeRh or self.BodeOut:
            x = Data
        else:
            x = Data / IVGainAC

        if self.RemoveDC:
            print 'RemoveDC'
            for chk, chi, in sorted(ChIndexes.iteritems()):
                Data[:, chi[1]] = Data[:, chi[1]] - np.mean(Data[:, chi[1]])

        FFTconf = self.BodeSignal.FFTconfs[Ind]

        (samps, inch) = x.shape
        Gm = np.ones((len(FFTconf.Finds),
                      inch))*np.complex(np.nan)

        for chk, chi, in sorted(ChIndexes.iteritems()):
            Out = self.BodeSignal.CalcFFT(Data=x[:, chi[1]],
                                          Ind=Ind)

#            Delay = self.AdqDelay*chi[0]  ## TODO check this time finer
            self.BodeSignal.GenSignal(Ind=Ind, Delay=self.AdqDelay)

            if self.BodeRh:
                Iin = -self.BodeSignal.FFTAmps/self.BodeRhardware
                gm = Out/Iin
                Gm[:, chi[1]] = gm
            else:
                Gm[:, chi[1]] = Out/self.BodeSignal.FFTAmps

        return Gm

###############################################################################
#####
###############################################################################


class DataProcess(ChannelsConfig, FFTBodeAnalysis):
    # PSD Config
    PSDnFFT = 2**17
    PSDFs = 30e3
    PSDnAvg = 5

    # Variables list
    DCFs = 1000
    DCnSamps = 1000
    IVGainDC = 10e3
    IVGainAC = 1e6
    IVGainGate = 2.2e6
    DevCondition = 5e-8
    PSDDuration = None
    GenTestSig = None
    ContSig = None
    OldConfig = None
    GmH = None
    Gm = None
    iConf = 0
    SeqIndex = 0

    # Events list
    EventBiasDone = None
    EventBiasAttempt = None
    EventBodeDone = None
    EventPSDDone = None
    EventAcDataAcq = None
    EventContAcDone = None
    EventContDcDone = None
    EventContGateDone = None
    EventSetBodeLabel = None

    def ClearEventsCallBacks(self):
        self.DCDataDoneEvent = None
        self.DCDataEveryNEvent = None
        self.ACDataDoneEvent = None
        self.ACDataEveryNEvent = None
        self.GateDataDoneEvent = None
        self.GateDataEveryNEvent = None

    # DC
    ####
    def GetBiasCurrent(self, Vds, Vgs):
        self.ClearEventsCallBacks()
        self.SetBias(Vds, Vgs)
        self.DCDataDoneEvent = self.CalcBiasData
        if self.GateChannelIndex is not None:
            self.GateDataDoneEvent = self.CalcGateData
        self.ReadChannelsData(Fs=self.DCFs,
                              nSamps=self.DCnSamps,
                              EverySamps=self.DCnSamps)

    def GetContinuousCurrent(self, Fs, Refresh, GenTestSig):
        print 'Cont GetContinuousCurrent'
        self.ClearEventsCallBacks()
        self.DCDataEveryNEvent = self.CalcDcContData
        self.ACDataEveryNEvent = self.CalcAcContData
        if self.GateChannelIndex is not None:
            self.GateDataEveryNEvent = self.CalcGateContData
        if GenTestSig:
            print 'CreateSignal'
            self.GenTestSig = GenTestSig
            print 'DataProcess SetContSignal'
            signal, _ = self.ContSig.GenSignal()
            self.SetContSignal(Signal=signal,
                               nSamps=signal.size)
        self.Inputs.ReadContData(Fs=Fs,
                                 EverySamps=Fs*Refresh)

    def CalcBiasData(self, Data):
        #  data = Data[1:, :]

        data = Data
        r, c = data.shape
        x = np.arange(0, r)
        mm, oo = np.polyfit(x, data, 1)
        Dev = np.abs(np.mean(mm))
        print 'DataProcess Attempt ', Dev
        if self.EventBiasAttempt:
            Ids = (data-self.BiasVd)/self.IVGainDC
            if not self.EventBiasAttempt(Ids,
                                         x*(1/np.float32(self.Inputs.Fs)),
                                         Dev):
                return  # Cancel execution

        if (Dev < self.DevCondition):
            Ids = (oo-self.BiasVd)/self.IVGainDC
            if self.EventBiasDone:
                self.EventBiasDone(Ids)
            return

        # try next attempt
        self.ReadChannelsData(Fs=self.DCFs,
                              nSamps=self.DCnSamps,
                              EverySamps=self.DCnSamps)

    def CalcGateData(self, Data):
        data = Data[1:, :]
        r, c = data.shape
        x = np.arange(0, r)
        mm, oo = np.polyfit(x, data, 1)
#        Dev = np.abs(np.mean(mm))
#        if (Dev < self.DevCondition):
#            if np.abs(mm) < self.DevCondition:
#                print 'Gate slope ', mm
#            else:
#                print 'WARNING !!! Gate slope ', mm
        Igs = oo/self.IVGainGate
        if self.EventGateDone:
            self.EventGateDone(Igs)
        return

    # Continuous acquisition
    ####
    def CalcDcContData(self, Data):
        print 'DataProcess CalcDCContData'
        print Data.shape
        Ids = (Data-self.BiasVd)/self.IVGainDC
        if self.EventContDcDone:
            self.EventContDcDone(Ids)

    def CalcAcContData(self, Data):
        print 'DataProcess CalcACContData'
        print Data.shape
        Ids = (Data-self.BiasVd)/self.IVGainAC
        if self.EventContAcDone:
            self.EventContAcDone(Ids)

    def CalcGateContData(self, Data):
        print 'DataProcess CalcGateContData'
        Igs = (Data-5e-3)/self.IVGainGate
#        data = Data[1:, :]
#        r, c = data.shape
#        x = np.arange(0, r)
#        mm, oo = np.polyfit(x, data, 1)
#        Igs = oo/self.IVGainGate
        if self.EventContGateDone:
            self.EventContGateDone(Igs)

    # AC
    ####
    def GetSeqBode(self, SeqConf):
        print 'DataProcess GetSeqBode'
        FFTconf = self.BodeSignal.FFTconfs[self.iConf]

        if SeqConf:
            print 'InitInptuns', SeqConf
            self.InitInputs(**SeqConf)

        signal, _ = self.BodeSignal.GenSignal(Ind=self.iConf)
        self.SetContSignal(Signal=signal, nSamps=signal.size)

        if self.EventSetBodeLabel:
            self.EventSetBodeLabel(Vpp=self.BodeSignal.Vpp)

        print 'Acquire Bode data for ', self.BodeSignal.BodeDuration[self.iConf], ' Seconds'

        self.ReadChannelsData(Fs=FFTconf.Fs,
                              nSamps=FFTconf.nFFT*self.BodeSignal.nAvg,
                              EverySamps=FFTconf.nFFT)

    def NextChannelAcq(self):
        print 'DataProcess NextChannelAcq'
        FFTconf = self.BodeSignal.FFTconfs[self.iConf]

        SeqConf = self.InitConfig.copy()
        if self.InitConfig['Configuration'] == 'Both':
            SeqConf['Configuration'] = 'AC'

        if self.InitConfig['GateChannel'] is not None:
            SeqConf['GateChannel'] = None

        if FFTconf.AcqSequential is True:

            if self.SeqIndex <= len(self.InitConfig['Channels']) - 1:
                Channel = [sorted(self.InitConfig['Channels'])[self.SeqIndex], ]
                print 'Channel -->', Channel
                SeqConf['Channels'] = Channel
                self.SeqIndex += 1

            else:
                print 'End Seq'
                self.SeqIndex = 0
                if len(self.BodeSignal.FFTconfs) > 1 and self.iConf == 0:
                    self.iConf += 1
                else:
                    self.iConf = 0
                    if self.OldConfig:
                        self.InitInputs(**self.OldConfig)

                    if self.EventBodeDone:
                        self.EventBodeDone(self.Gm, self.BodeSignal.Fsweep)
                    return

        else:
            self.iConf = 0
            if self.OldConfig:
                self.InitInputs(**self.OldConfig)

            if self.EventBodeDone:
                self.EventBodeDone(self.Gm, self.BodeSignal.Fsweep)
            return

        self.GetSeqBode(SeqConf)

    def GetBode(self):
        self.OldConfig = self.InitConfig.copy()
        if self.InitConfig['Configuration'] == 'Both':
            print 'Config Both'
            self.OldConfig = self.InitConfig.copy()
            conf = self.InitConfig.copy()
            conf['Configuration'] = 'AC'
            self.InitInputs(**conf)

        self.ClearEventsCallBacks()
        self.ACDataDoneEvent = self.CalcBodeData
        self.ACDataEveryNEvent = self.EventDataAcq

        ##
        if len(self.BodeSignal.FFTconfs) > 1:
            self.GmH = np.ones((len(self.BodeSignal.FFTconfs[0].Freqs),
                                len(self.ACChannelIndex.items())))*np.complex(np.nan)

        self.Gm = np.ones((len(self.BodeSignal.Fsweep),
                           len(self.ACChannelIndex.items())))*np.complex(np.nan)
        ##
        self.NextChannelAcq()

    def CalcBodeData(self, Data):
        print 'DataProcess CalcBodeData_Data'

        FFTconf = self.BodeSignal.FFTconfs[self.iConf]

        if FFTconf.AcqSequential is True:
            GmSeq = self.CalcBode(Data=Data,
                                  Ind=self.iConf,
                                  IVGainAC=self.IVGainAC,
                                  ChIndexes=self.ACChannelIndex)

            if len(self.BodeSignal.FFTconfs) > 1:
                self.GmH[:, self.SeqIndex - 1] = GmSeq[:, 0]
            else:
                self.Gm[:, self.SeqIndex - 1] = GmSeq[:, 0]

            self.NextChannelAcq()

            return

        else:
            print self.iConf
            GmL = self.CalcBode(Data=Data,
                                Ind=self.iConf,
                                IVGainAC=self.IVGainAC,
                                ChIndexes=self.ACChannelIndex)

            self.Gm = np.vstack((GmL, self.GmH))

        self.NextChannelAcq()

    def GetPSD(self):
        self.ClearEventsCallBacks()
        self.ACDataEveryNEvent = self.EventDataAcq
        if not self.PSDDuration:
            self.PSDDuration = self.PSDnFFT*self.PSDnAvg*(1/self.PSDFs)
        print 'DataProcess Acquire PSD data for ', self.PSDDuration, 'seconds'
        self.ACDataDoneEvent = self.CalcPSDData
        self.ReadChannelsData(Fs=self.PSDFs,
                              nSamps=self.PSDnFFT*self.PSDnAvg,
                              EverySamps=self.PSDnFFT)

    def CalcPSDData(self, Data):
        data = Data/self.IVGainAC

        ff, psd = signal.welch(x=data,
                               fs=self.PSDFs,
                               window='hanning',
                               nperseg=self.PSDnFFT,
                               scaling='density', axis=0)
        if self.EventPSDDone:
            self.EventPSDDone(psd, ff, data)

    def EventDataAcq(self, Data):
        print 'DataProcess EventAcDataAcq'
        r, c = Data.shape
        x = np.arange(0, r)
        print self.Inputs.Fs
        if self.EventAcDataAcq:

            self.EventAcDataAcq(Data/self.IVGainAC,
                                x*(1/np.float32(self.Inputs.Fs)))


###############################################################################
#####
###############################################################################


class Charact(DataProcess):
    # Sweep points
    SwVdsVals = None
    SwVdsInd = None
    SwVgsVals = None
    SwVgsInd = None
    DevACVals = None

    # Status vars
    CharactRunning = False
    RunningConfig = False
    GateChannel = None

    # events list
    EventCharSweepDone = None
    EventCharBiasDone = None
    EventCharBiasAttempt = None
    EventCharAcDataAcq = None
    EventFFTDone = None

    # Neo Record
    ContRecord = None

    def InitSweep(self, VdsVals, VgsVals, PSD=False, Bode=False):
        print 'Charact InitSweep'
        self.SwVgsVals = VgsVals
        self.SwVdsVals = VdsVals
        self.Bode = Bode
        self.PSD = PSD
        self.SwVgsInd = 0
        self.SwVdsInd = 0

        self.EventBiasAttempt = self.BiasAttemptCallBack
        self.EventBiasDone = self.BiasDoneCallBack
        self.EventGateDone = self.GateDoneCallBack
        if self.Bode:
            self.EventBodeDone = self.BodeDoneCallBack
            self.BodeSignal.EventFFTDebug = self.EventFFT  # TODO implement if condition
        if self.PSD:
            self.EventPSDDone = self.PSDDoneCallBack
        self.EventAcDataAcq = self.AcAcqCallBack

        self.CharactRunning = True

        # Init Dictionaries
        self.InitDictionaries()
        self.ApplyBiasPoint()

    def InitDictionaries(self):
        # DC dictionaries
        if self.GateChannelIndex is None:
            Gate = False
        else:
            Gate = True

        self.DevDCVals = PyData.InitDCRecord(nVds=self.SwVdsVals,
                                             nVgs=self.SwVgsVals,
                                             ChNames=self.ChNamesList,
                                             Gate=Gate)
        # AC dictionaries
        if self.Bode or self.PSD:
            if self.PSD:
                Fpsd = np.fft.rfftfreq(self.PSDnFFT, 1/self.PSDFs)
            else:
                Fpsd = np.array([])

            if self.Bode:
                print 'Bode dict'
                nFgm = self.BodeSignal.Fsweep
            else:
                nFgm = np.array([])

            self.DevACVals = PyData.InitACRecord(nVds=self.SwVdsVals,
                                                 nVgs=self.SwVgsVals,
                                                 nFgm=nFgm,
                                                 nFpsd=Fpsd,
                                                 ChNames=self.ChNamesList)

    def ApplyNextBias(self):
        print 'Charact ApplyNextBias'
        if self.SwVdsInd < len(self.SwVdsVals)-1:
            self.SwVdsInd += 1
        else:
            self.SwVdsInd = 0
            if self.SwVgsInd < len(self.SwVgsVals)-1:
                self.SwVgsInd += 1
            else:
                self.SwVgsInd = 0

                if self.EventCharSweepDone:
                    self.EventCharSweepDone(self.DevDCVals, self.DevACVals)
                return

        if self.CharactRunning:
            self.ApplyBiasPoint()
        else:
            self.StopCharac()

    def ApplyBiasPoint(self):
        print 'Charact ApplyBiasPoint'
        self.GetBiasCurrent(Vds=self.SwVdsVals[self.SwVdsInd],
                            Vgs=self.SwVgsVals[self.SwVgsInd])

        if self.EventSetLabel:
            self.EventSetLabel(self.SwVdsVals[self.SwVdsInd],
                               self.SwVgsVals[self.SwVgsInd])

    def InitContMeas(self, Vds, Vgs, Fs, Refresh,
                     RecDC=True, RecAC=True, RecGate=False, GenTestSig=False):
        print 'Charact InitContMeas'
        #  Init Neo record
        out_seg = neo.Segment(name='NewSeg')

        if RecAC:
            self.EventContAcDone = self.ContAcDoneCallback
            for chk, chi, in sorted(self.ACChannelIndex.iteritems()):
                name = chk + '_AC'
                sig = neo.AnalogSignal(signal=np.empty((0, 1), float),
                                       units=pq.V,
                                       t_start=0*pq.s,
                                       sampling_rate=Fs*pq.Hz,
                                       name=name)
                out_seg.analogsignals.append(sig)

        if RecDC:
            self.EventContDcDone = self.ContDcDoneCallback
            for chk, chi, in sorted(self.DCChannelIndex.iteritems()):
                name = chk + '_DC'
                sig = neo.AnalogSignal(signal=np.empty((0, 1), float),
                                       units=pq.V,
                                       t_start=0*pq.s,
                                       sampling_rate=Fs*pq.Hz,
                                       name=name)
                out_seg.analogsignals.append(sig)

        if RecGate:
            print 'GateCont ok'
            self.EventContGateDone = self.ContGateDoneCallback
            for chk, chi, in sorted(self.GateChannelIndex.iteritems()):
                name = chk + '_Gate'
                sig = neo.AnalogSignal(signal=np.empty((0, 1), float),
                                       units=pq.V,
                                       t_start=0*pq.s,
                                       sampling_rate=Fs*pq.Hz,
                                       name=name)
                out_seg.analogsignals.append(sig)

        self.ContRecord = NeoRecord(Seg=out_seg, UnitGain=1)

        #  Lauch adquisition
        self.SetBias(Vds=Vds, Vgs=Vgs)
        self.GetContinuousCurrent(Fs=Fs,
                                  Refresh=Refresh,
                                  GenTestSig=GenTestSig)
        self.CharactRunning = True

    def BiasAttemptCallBack(self, Ids, time, Dev):
        if self.EventCharBiasAttempt:
            self.EventCharBiasAttempt(Ids, time, Dev)

        return self.CharactRunning

    def AcAcqCallBack(self, Ids, time):
        print 'Charact AcAcqCallBack'
        if self.EventCharAcDataAcq:
            self.EventCharAcDataAcq(Ids, time)
        else:
            self.StopCharac()
#        return self.CharactRunning

    def BiasDoneCallBack(self, Ids):
        print 'Charact BiasDoneCallBack'

        for chn, inds in self.DCChannelIndex.iteritems():
            self.DevDCVals[chn]['Ids'][self.SwVgsInd,
                                       self.SwVdsInd] = Ids[inds[1]]

        if self.EventCharBiasDone:
            self.EventCharBiasDone(self.DevDCVals)
        # Measure AC Data
        if self.CharactRunning:
            if self.Bode:
                self.GetBode()
            elif self.PSD:
                self.GetPSD()
            else:
                self.ApplyNextBias()

    def GateDoneCallBack(self, Igs):
        print 'Charact GateDoneCallBack'
        for chn, inds in self.GateChannelIndex.iteritems():
            self.DevDCVals['Gate']['Ig'][self.SwVgsInd, self.SwVdsInd] = Igs

    def BodeDoneCallBack(self, Gm, SigFreqs):
        print 'Charact BodeDoneCallBack'
        for chn, inds in self.ACChannelIndex.iteritems():
            self.DevACVals[chn]['gm']['Vd{}'.format(self.SwVdsInd)][
                    self.SwVgsInd] = Gm[:, inds[1]]
            self.DevACVals[chn]['Fgm'] = SigFreqs

        print Gm.shape, SigFreqs.shape
        if self.EventCharACDone:
            self.EventCharACDone(self.DevACVals)

        if self.CharactRunning:
            if self.PSD:
                self.GetPSD()
            else:
                self.ApplyNextBias()
        else:
            self.StopCharac()

    def PSDDoneCallBack(self, psd, ff, data):
        for chn, inds in self.ACChannelIndex.iteritems():
            self.DevACVals[chn]['PSD']['Vd{}'.format(self.SwVdsInd)][
                    self.SwVgsInd] = psd[:, inds[1]]
            self.DevACVals[chn]['Fpsd'] = ff
        print psd.shape, ff.shape, 'Charact PsdDoneCallback'
        if self.EventCharACDone:
            self.EventCharACDone(self.DevACVals)

        self.ApplyNextBias()

    def EventFFT(self, FFT):
        if self.EventFFTDone:
            self.EventFFTDone(FFT)

    def ContDcDoneCallback(self, Ids):
        print 'Charact Continuous Dc Data Done Callback'
        for chk, chi, in self.DCChannelIndex.iteritems():
            newvect = Ids[:, chi[1]].transpose()
            self.ContRecord.AppendSignal(chk + '_DC', newvect[:, None])

        tstop = self.ContRecord.Signal(ChName=chk + '_DC').t_stop

        if (self.EventContAcDone is None) and (self.EventContGateDone is None):
            if self.EventContinuousDone:
                self.EventContinuousDone(tstop)

    def ContGateDoneCallback(self, Igs):
        print 'Charact Continuous Gate Data Done Callback'
        for chk, chi, in self.GateChannelIndex.iteritems():
            newvect = Igs[:, chi[1]].transpose()
            self.ContRecord.AppendSignal(chk + '_Gate', newvect[:, None])

        tstop = self.ContRecord.Signal(ChName=chk + '_Gate').t_stop

        if self.EventContAcDone is None:
            if self.EventContinuousDone:
                self.EventContinuousDone(tstop)

    def ContAcDoneCallback(self, Ids):
        print 'Charact Continuous Ac Data Done Callback'
        for chk, chi, in self.ACChannelIndex.iteritems():
            newvect = Ids[:, chi[1]].transpose()
            self.ContRecord.AppendSignal(chk + '_AC', newvect[:, None])

        tstop = self.ContRecord.Signal(ChName=chk + '_AC').t_stop

        if self.EventContinuousDone:
            self.EventContinuousDone(tstop)

    def StopCharac(self):
        print 'STOP'
        self.CharactRunning = False
#        self.Inputs.ClearTask()
        if self.ContRecord:
            self.Inputs.StopContData()
            if self.GenTestSig:
                self.VgOut.StopTask()
                self.VgOut = None

###############################################################################
#####
###############################################################################


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
                sl.DispName = 'DC'

            if sign.endswith('_AC'):
                sl.Position = 1
                sl.Color = cmap.to_rgba(ind)
                sl.DispName = sign

            if sign.endswith('_Gate'):
                sl.Position = 2
                sl.DispName = sign

            sl.SigName = sign

#            sl.Position = ind

#            if sign.endswith('_DC'):
#                sl.FiltType = ('lp', )
#                sl.FiltOrder = (2, )
#                sl.FiltF1 = (1, )

            sl.OutType = 'V'
            slots.append(sl)

        #  Init Plot figures
        self.PltRecs = PlotRecord()
        self.PltRecs.CreateFig(slots)
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
        uipath = os.path.join(os.path.dirname(__file__), 'PyFETCharactGui.ui')
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
        self.Charac = Charact(Channels=Channels,
                              GateChannel=GateChannel,
                              Configuration=Config)

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
