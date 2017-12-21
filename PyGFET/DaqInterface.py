# -*- coding: utf-8 -*-
"""
@author: Anton Guimerà
@version: 0.1b

Revsion history

- 0.1b -- First version

"""

import PyDAQmx as Daq
import numpy as np
from ctypes import byref, c_int32
import matplotlib.pyplot as plt
from scipy import signal
import math
from scipy import interpolate


###############################################################################
##### TODO Callback 
###############################################################################

#class ReadAnalogCallBack(Daq.Task):
#    '''
#    Class to read data from Daq card
#
#    TODO - Implement the callback option to read data
#    '''
#    def __init__(self, InChans, Range=5.0):
#        Daq.Task.__init__(self)
#
#        self.Channels=InChans
#
#        for Ch in self.Channels:
#            self.CreateAIVoltageChan('Dev1/{}'.format(Ch),"",
#                                     Daq.DAQmx_Val_RSE,
#                                     -Range,Range,
#                                     Daq.DAQmx_Val_Volts,None)
#
#    def ReadData(self, Fs = 1000, nSamps = 1000): ### Add function to call
#        '''
#        Return data
#        '''
#
#        read = c_int32()
#        data = np.zeros((nSamps,len(self.Channels)))
#        TimeOut = nSamps * (1/Fs) + 1
#
#        self.CfgSampClkTiming("", Fs, Daq.DAQmx_Val_Rising,
#                              Daq.DAQmx_Val_FiniteSamps, nSamps)
#        
##        self.AutoRegisterEveryNSamplesEvent(DAQmx_Val_Acquired_Into_Buffer,1000,0)
#        
#        self.StartTask()
#
#        ### En el callback
#        self.ReadAnalogF64(nSamps, TimeOut, Daq.DAQmx_Val_GroupByScanNumber,
#                           data, data.size, byref(read), None)
#      
#        
#        self.StopTask()
#
#        return data



###############################################################################
#####
###############################################################################
class ReadAnalog(Daq.Task):
    '''
    Class to read data from Daq card

    TODO - Implement the callback option to read data
    '''
    def __init__(self, InChans, Range=5.0):
        Daq.Task.__init__(self)

        self.Channels=InChans

        for Ch in self.Channels:
            self.CreateAIVoltageChan('Dev1/{}'.format(Ch),"",
                                     Daq.DAQmx_Val_RSE,
                                     -Range,Range,
                                     Daq.DAQmx_Val_Volts,None)

    def ReadData(self, Fs = 1000, nSamps = 1000):
        '''
        Return data
        '''

        read = c_int32()
        data = np.zeros((nSamps,len(self.Channels)))
        TimeOut = nSamps * (1/Fs) + 1

        self.CfgSampClkTiming("", Fs, Daq.DAQmx_Val_Rising,
                              Daq.DAQmx_Val_FiniteSamps, nSamps)

        self.StartTask()
        self.ReadAnalogF64(nSamps, TimeOut, Daq.DAQmx_Val_GroupByScanNumber,
                           data, data.size, byref(read), None)
        self.StopTask()

        return data

###############################################################################
#####
###############################################################################
class WriteAnalog(Daq.Task):
    '''
    Class to write data to Daq card
    '''
    def __init__(self, Channel):
        Daq.Task.__init__(self)
        self.CreateAOVoltageChan('Dev1/{}'.format(Channel), "",
                                 -5.0, 5.0, Daq.DAQmx_Val_Volts, None)
    def SetVal(self, value):
        self.StartTask()
        self.WriteAnalogScalarF64(1,10.0,value,None)
        self.StopTask()
    def SetSignal(self, Signal):
        read = c_int32()
        nSamps = Signal.size
        self.CfgSampClkTiming('ai/SampleClock', 1, Daq.DAQmx_Val_Rising,
                              Daq.DAQmx_Val_FiniteSamps, nSamps)

        self.WriteAnalogF64(nSamps,False,1000,Daq.DAQmx_Val_GroupByScanNumber,Signal,byref(read),None)
        self.CfgDigEdgeStartTrig('ai/StartTrigger', Daq.DAQmx_Val_Rising)
        self.StartTask()

###############################################################################
#####
###############################################################################
class DCBias():
    '''
    Class with to analog outputs to set de bias point
    '''

    def __init__(self, ChVs='ao1', ChVd='ao0'):
        self.Vgs = WriteAnalog(ChVs)
        self.Vds = WriteAnalog(ChVd)
    def SetBias(self, Vds, Vgs):
        self.Vgs.SetVal(-Vgs)
        self.Vds.SetVal(Vds)
    def __del__(self):
        self.Vds.ClearTask()
        self.Vgs.ClearTask()

###############################################################################
#####
###############################################################################
class DCCharacterization():
    '''

    '''

    DaqDCChannels = ('ai1', 'ai3', 'ai4', 'ai5', 'ai6', 'ai7',
           'ai17', 'ai19', 'ai20', 'ai21', 'ai22', 'ai23')
    Fs = 1000
    nSamps = 1000
    IVGainDC = 10e3
    DevCondition = 5e-8
    IVGainGate = 3.5e6

    def __init__(self, ChDrains=None, ChGate=None, ChVs='ao1', ChVd='ao0',DCGianCal=None):
        self.Bias = DCBias(ChVs=ChVs, ChVd=ChVd)

        if ChDrains:
            self.DCin = ReadAnalog(InChans=ChDrains)
        else:
            self.DCin = ReadAnalog(InChans=self.DaqDCChannels)

        if ChGate:
            self.GateIn = ReadAnalog(InChans=ChGate)
        else:
            self.GateIn = None

        if DCGianCal:
            self.IVGainDC=1

    def GetBiasCurrent(self, Vds, Vgs):
        '''
        return Ids,Igs
        '''

        print 'Set Bias Vgs ->', Vgs, 'Vds -> ',Vds
        self.Bias.SetBias(Vds, Vgs)

        data = self.DCin.ReadData(Fs=self.Fs, nSamps=self.nSamps)
        r,c= data.shape
        x = np.arange(0,r)
        mm, oo = np.polyfit(x,data,1)
        i=0

        while (np.abs(np.mean(mm))>self.DevCondition):
            data = self.DCin.ReadData(Fs=self.Fs, nSamps=self.nSamps)
            mm, oo = np.polyfit(x,data,1)
            print 'Attempt ', i ,np.abs(np.mean(mm))
            i+=1

        Vd = Vds -Vgs;
        Ids = (oo-Vd)/self.IVGainDC

        if self.GateIn:
            data = self.GateIn.ReadData(Fs=self.Fs, nSamps=self.nSamps)
            mm, oo = np.polyfit(x,data,1)

            if np.abs(mm)<self.DevCondition:
                print 'Gate slope ', mm
            else:
                print 'WARNING !!! Gate slope ', mm
            Igs = oo/self.IVGainGate
        else:
            Igs = np.NaN

        return Ids,Igs,data

    def __del__(self):
        del(self.Bias)
        self.DCin.ClearTask()


###############################################################################
#####
###############################################################################
class ACCharacterization():
    '''
    Implementation of PSD and Bode measurements

    TODO Fix the last point of signal generation, it will be 0
    '''
    DaqACChannels = ('ai9', 'ai11', 'ai12', 'ai13', 'ai14', 'ai15',
                     'ai25', 'ai27', 'ai28', 'ai29', 'ai30', 'ai31')
    DaqOutChannel = 'ao2'
    IVGainAC = 1e6
    IVGainACCal = None
    RHardWare = 150e3

    SigFreqMin = 0.08
    SigFreqMax = 15e3
    SignFreqs = 50
    SigFs = 50e3
    SigPoints = 2**21
    SigAmp = 0.1e-3
    
    PSDnAvg=10 
    PSDnFFT=2**15 
    PSDFs = 50e3

    def __init__(self,CalFile=None):

        self.ACin = ReadAnalog(InChans=self.DaqACChannels)
        self.Gate = WriteAnalog(Channel=self.DaqOutChannel)


        self.CreateSignal()
        self.Gate.SetVal(0)        

        if CalFile:
            f={}
            self.IVGainACCal=np.ones((len(self.SigFreqs),len(self.DaqACChannels)))*np.NaN
            for ich,ch in enumerate(self.DaqACChannels):
                f[ch]=interpolate.interp1d(self.SigFreqs,CalFile[ch])
                fun=f[ch]
                self.IVGainACCal[:,ich]=fun(np.logspace(math.log(self.SigFreqs[0],10),math.log(self.SigFreqs[-1],10), num=len(self.SigFreqs), endpoint=True))#interpolacioper cada self.SigFreqs
                    
                
            
    def CreateSignal(self, Fs=None, Points=None, Freqs=None):

        if Fs: self.SigFs = Fs
        if Points: self.SigPoints = Points

        self.SigTs = 1/self.SigFs

        nmin = (self.SigFreqMin*(self.SigPoints))/self.SigFs
        nmax = (self.SigFreqMax*(self.SigPoints))/self.SigFs

        if Freqs:
            self.SigFreqs = Freqs
        else:
            self.SigFreqs = (self.SigFs/self.SigPoints)*np.unique(np.round(
                np.logspace(np.log10(nmin),np.log10(nmax),self.SignFreqs),0))

        self.SigT = np.arange(0,self.SigPoints * self.SigTs,self.SigTs)
        self.Sig = np.zeros(self.SigT.size)
        for f in np.nditer(self.SigFreqs):
            self.Sig += self.SigAmp*np.sin(f*2*np.pi*self.SigT)

        self.SigFFT = 2*np.fft.rfft(self.Sig,self.SigPoints)/self.SigPoints
        self.SigFF = np.fft.rfftfreq(self.SigPoints,self.SigTs)

        self.SigFF = [np.round(x,5) for x in self.SigFF]
        self.SigFreqs = [np.round(x,5)  for x in self.SigFreqs]
        self.SigFF = np.array(self.SigFF)
        self.SigFreqs = np.array(self.SigFreqs)

        self.SigFInds = np.where(np.in1d(self.SigFF,self.SigFreqs)==True)
        self.SigIn = self.SigFFT[self.SigFInds]

        self.Sig[-1]=0

    def GetBode(self, HardWare=False):
        '''
        return Gm, SigFreqs, outFFT
        '''
        print 'Acquire Bode data for ', self.SigPoints*self.SigTs, ' Seconds'
        self.Gate.SetSignal(Signal=self.Sig)
        Data = self.ACin.ReadData(Fs=self.SigFs ,nSamps=self.SigPoints)
        self.Gate.StopTask()

        outFFT = 2*np.fft.rfft(Data,self.SigPoints,axis=0)/self.SigPoints
        Out = outFFT[self.SigFInds]

#        self.Gate.SetVal(0) #TODO check this error -- SOlved by self.Sig[-1]=0 

        if HardWare:
            Iin = self.SigIn/self.RHardWare
            Gain = Out/Iin[:,None]
            return Gain, self.SigFreqs, outFFT
        else:
#            if self.IVGainACCal.any():
#                print "Corrected"
#                Iin = Out/self.IVGainACCal
#                Gm = Iin/self.SigIn[:,None]
#                return Gm, self.SigFreqs, outFFT            
#            else:      
            
            Iin = Out/self.IVGainAC
            Gm = Iin/self.SigIn[:,None]
            return Gm, self.SigFreqs, outFFT

            

    def GetPSD(self, nAvg=None , nFFT=None, Fs = None, Gain=True):
        '''
        return psd, ff, data
        '''
        
        if not nAvg:
            nAvg = self.PSDnAvg
        
        if not nFFT:
            nFFT = self.PSDnFFT
            
        if not Fs:
            Fs = self.PSDFs
        
        nSamps = nFFT*nAvg
        print 'Acquire PSD data for ', nSamps*(1/Fs), ' Seconds'
        data = self.ACin.ReadData(Fs=Fs, nSamps=nSamps)
        
        if Gain:        
            data = data/self.IVGainAC

        ff,psd = signal.welch(x=data,fs=Fs, window='hanning',
                              nperseg=nFFT, scaling='density',axis=0)

        return psd, ff, data

    def __del__(self):
        self.ACin.ClearTask()
        self.Gate.ClearTask()


#       self.StopTask()

if __name__ == '__main__':

#########################################################################
#    Signal genaration test
#    Connect Gate signal to one AC input

#    plt.close('all')
#    plt.ion()
#
#    ACchar = ACCharacterization()


#    Gain, SigFreqs, OutSig = ACchar.GetBode(HardWare=True)
#
#    InSig = ACchar.SigFFT
#
#    fig1,(ax1,ax2) = plt.subplots(2,1,sharex=True)
#    ax1.semilogx(ACchar.SigFF, np.abs(InSig),'ro')
#    ax2.semilogx(ACchar.SigFF, np.angle(InSig)*180/np.pi,'ro')
#    ax1.semilogx(ACchar.SigFF, np.abs(OutSig),'b-')
#    ax2.semilogx(ACchar.SigFF, np.angle(OutSig)*180/np.pi,'b-')
#
#    fig1,(ax1,ax2) = plt.subplots(2,1,sharex=True)
#    ax1.semilogx(SigFreqs, np.abs(ACchar.SigIn),'ro')
#    ax2.semilogx(SigFreqs, np.angle(ACchar.SigIn)*180/np.pi,'ro')
#    ax1.semilogx(SigFreqs, np.abs(OutSig[ACchar.SigFInds]),'b-')
#    ax2.semilogx(SigFreqs, np.angle(OutSig[ACchar.SigFInds])*180/np.pi,'b-')
#
#    ax2.set_ylim(-180,-85)
#    ax1.set_ylim(ACchar.SigAmp-10e-6,ACchar.SigAmp+10e-6)
#
#    del(ACchar)

#########################################################################
#    Hardware Bode
#    Connect Gate signal to 150e3 resistance

#    plt.close('all')
#    plt.ion()
#
#    ACchar = ACCharacterization()
#
#
#    Gain, SigFreqs, OutSig = ACchar.GetBode(HardWare=True)
#
#    fig1,(ax1,ax2) = plt.subplots(2,1,sharex=True)
#    ax1.loglog(SigFreqs, np.abs(Gain))
#    ax2.semilogx(SigFreqs, np.angle(Gain)*180/np.pi)
#    ax1.grid()
#    ax2.grid()
#
#    del(ACchar)


####################################################################
######### Test DC Noise Charact

    plt.close('all')
    plt.ion()

    DCchar = DCCharacterization(ChGate=('ai0',))
    ACchar = ACCharacterization()

    nVgs = np.linspace(0.1,0.5,5)
    DCchar.DevCondition=5e-7

    fig1, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)


    Curr = np.ones((len(nVgs),12))*np.NaN
    Res = np.ones((len(nVgs),12))*np.NaN
    Gate = np.ones(len(nVgs))*np.NaN
    i = 0
    for Vgs in nVgs:
        Ids,Rds,Ig = DCchar.GetBiasCurrent(0.1, Vgs)

        Curr[i,:] = Ids
        Res[i,:] = Rds
        Gate[i] = Ig
        i += 1
        ax1.plot(nVgs,Curr)
        ax2.plot(nVgs,Res)
        ax3.plot(nVgs,Gate)
        plt.pause(0.1)

        Gm, SigFreqs, OutSig = ACchar.GetBode()

        fig2,(ax5,ax6) = plt.subplots(2,1,sharex=True)
        ax5.loglog(SigFreqs, np.abs(Gm))
        ax6.semilogx(SigFreqs, np.angle(Gm)*180/np.pi)
        ax5.grid()
        ax6.grid()

#        psd,ff,data = ACchar.GetPSD()
#        ax3.loglog(ff,np.sqrt(psd))
#        ax4.plot(data)
#        plt.pause(0.05)


    DCchar.Bias.SetBias(0,0)

    del(DCchar)
    del(ACchar)



