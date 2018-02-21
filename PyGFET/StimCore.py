# -*- coding: utf-8 -*-
"""
Created on Fri Feb 09 10:58:53 2018

@author: Javier
"""

import sys
import ctypes
from PyGFET.RecordCore import NeoRecord
import PyDAQmx as Daq
from ctypes import byref, c_int32
import numpy as np
import neo
import quantities as pq

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

        print 'ReadAnalog Init'
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
        print 'ReadAnalog GetDevName'
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

        if not self.ContSamps:  # TODO check why stack here
            self.data = np.vstack((self.data, data))

#        print data.size, self.data.shape

        if self.EveryNEvent:
            self.EveryNEvent(data)

    def DoneCallback(self, status):
        print 'ReadAnalog Done'
        self.StopTask()
        self.UnregisterEveryNSamplesEvent()

        if self.DoneEvent:
            self.DoneEvent(self.data[1:, :])  # TODO check why 1:

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
    aiChannels = {'Ch01': ('ai8',),
                  'Ch02': ('ai9',),
                  'Ch03': ('ai10',),
                  'Ch04': ('ai11',),
                  'Ch05': ('ai12',),
                  'Ch06': ('ai13',),
                  'Ch07': ('ai14',),
                  'Ch08': ('ai15',),
                  'Ch09': ('ai24',),
                  'Ch10': ('ai25',),
                  'Ch11': ('ai26',),
                  'Ch12': ('ai27',),
                  'Ch13': ('ai28',),
                  'Ch14': ('ai29',),
                  'Ch15': ('ai30',),
                  'Ch16': ('ai31',)}

# ChannelIndex = {'Ch01': (0-31, 0-15)}-->> {Chname: (input index, sort index)}
    DCChannelIndex = None

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
        print 'DelInputs'
        self.Inputs.ClearTask()

    def InitInputs(self, Channels, GateChannel=None,):
        print 'InitIputs'
        if self.Inputs is not None:
            self.DelInputs()

        InChans = []

        self.ChNamesList = sorted(Channels)
        self.DCChannelIndex = {}
        index = 0
        sortindex = 0
        for ch in sorted(Channels):
            InChans.append(self.aiChannels[ch][0])
            self.DCChannelIndex[ch] = (index, sortindex)
            index += 1
            sortindex += 1

        if GateChannel:
            self.GateChannelIndex = {GateChannel: (index, 0)}
            InChans.append(self.aiChannels[GateChannel][0])
        else:
            self.GateChannelIndex = None

        print 'Channels configurtation'
        print 'Channels ', len(self.ChNamesList)
        print 'ai list ->', InChans

        self.Inputs = ReadAnalog(InChans=InChans)
        # events linking
        self.Inputs.EveryNEvent = self.EveryNEventCallBack
        self.Inputs.DoneEvent = self.DoneEventCallBack

    def __init__(self, Channels, GateChannel=None, ChVg=None, ChVs=None,
                 ChVds=None, ChVsig='ao0'):
        self.InitConfig = {}
        self.InitConfig['Channels'] = Channels
        self.InitConfig['GateChannel'] = GateChannel

        self.InitInputs(Channels=Channels,
                        GateChannel=GateChannel)

        # Output Channels
#        self.VsOut = WriteAnalog((ChVs,))
#        self.VdsOut = WriteAnalog((ChVds,))
#        self.VgOut = WriteAnalog((ChVg,))
        self.VsigOut = WriteAnalog((ChVsig,))

    def SetBias(self, Vsig):
        print 'ChannelsConfig SetBias Vsig ->', Vsig
        self.VsigOut.SetVal(Vsig)
        self.Vin = Vsig

    def SetSignal(self, Signal, nSamps):
        if not self.VsigOut:
            self.VsigOut = WriteAnalog(('ao0',))
        self.VsigOut.DisableStartTrig()
        self.VsigOut.SetSignal(Signal=Signal,
                               nSamps=nSamps)

    def SetContSignal(self, Signal, nSamps):
        if not self.VsigOut:
            self.VsigOut = WriteAnalog(('ao0',))
        self.VsigOut.DisableStartTrig()
        self.VsigOut.SetContSignal(Signal=Signal,
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
        if self.VsigOut:
            self.VsigOut.StopTask()

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
        if self.VsigOut:
            self.VsigOut.ClearTask()
#        self.VdsOut.ClearTask()
#        self.VsOut.ClearTask()
        self.Inputs.ClearTask()


###############################################################################
#####
###############################################################################


class DataProcess(ChannelsConfig):

    # Variables list
    DCFs = 1000
    DCnSamps = 1000
    IVGainDC = 2
    Rds = 1.2e3
    Rharware = False
    DevCondition = 5e-8
    ContSig = None

    # Events list
#    EventBiasDone = None
#    EventBiasAttempt = None
#    EventBodeDone = None
#    EventPSDDone = None
#    EventAcDataAcq = None
#    EventContAcDone = None
    EventContDcDone = None
    GenTestSig = None
#    EventContGateDone = None
#    EventSetBodeLabel = None

    def ClearEventsCallBacks(self):
        self.DCDataDoneEvent = None
        self.DCDataEveryNEvent = None
        self.ACDataDoneEvent = None
        self.ACDataEveryNEvent = None
        self.GateDataDoneEvent = None
        self.GateDataEveryNEvent = None

    def GetContinuousCurrent(self, Fs, Refresh, GenTestSig):
        print 'Cont GetContinuousCurrent'
        self.ClearEventsCallBacks()
        self.DCDataEveryNEvent = self.CalcDcContData
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

    # Continuous acquisition
    ####
    def CalcDcContData(self, Data):
        print 'DataProcess CalcDCContData'
        print np.mean(Data)
        print self.Rds

#        if self.Rharware:
#            print self.Rds
#            Vload = Data * (self.Rds/(self.Rds+1e3))
#        else:
#            print 'no RDS'
#            Vload = Data + 2*self.Vin

        Vload = Data * (self.Rds/(self.Rds+1e3))
        print 'Vload ->', np.mean(Vload)
        Iload = Data * (1/(self.Rds+1e3))
        print 'Iload (Rds) ->', np.mean(Iload)
        ILoad = -2*self.Vin/1e3
        print 'Iload (Vin) ->', np.mean(ILoad)

        if self.EventContDcDone:
            self.EventContDcDone(Iload)


###############################################################################
#####
###############################################################################

class Charact(DataProcess):

    # Status vars
    CharactRunning = False

    # Neo Record
    ContRecord = None

    def InitContMeas(self, Vin, Fs, Refresh, RecDC=True, GenTestSig=False):
        print 'Charact InitContMeas'
        #  Init Neo record
        out_seg = neo.Segment(name='NewSeg')

        if RecDC:
            self.EventContDcDone = self.ContDcDoneCallback
            for chk, chi, in sorted(self.DCChannelIndex.iteritems()):
                name = chk
                sig = neo.AnalogSignal(signal=np.empty((0, 1), float),
                                       units=pq.V,
                                       t_start=0*pq.s,
                                       sampling_rate=Fs*pq.Hz,
                                       name=name)
                out_seg.analogsignals.append(sig)

        self.ContRecord = NeoRecord(Seg=out_seg, UnitGain=1)

        #  Lauch adquisition
        self.SetBias(Vsig=Vin)
        self.GetContinuousCurrent(Fs=Fs,
                                  Refresh=Refresh,
                                  GenTestSig=GenTestSig)
        self.CharactRunning = True

    def ContDcDoneCallback(self, Vload):
        print 'Charact Continuous Dc Data Done Callback'
        for chk, chi, in self.DCChannelIndex.iteritems():
            newvect = Vload[:, chi[1]].transpose()
            self.ContRecord.AppendSignal(chk, newvect[:, None])

        tstop = self.ContRecord.Signal(ChName=chk).t_stop

        if self.EventContinuousDone:
            self.EventContinuousDone(tstop)

    def StopCharac(self):
        print 'STOP'
        self.SetBias(Vsig=0)
        self.CharactRunning = False
#        self.Inputs.ClearTask()
        if self.ContRecord:
            self.Inputs.StopContData()
            if self.GenTestSig:
                self.VsigOut.StopTask()
                self.VsigOut = None
