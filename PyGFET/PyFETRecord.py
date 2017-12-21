#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 12:11:53 2017

@author: aguimera
"""

import neo
import numpy as np
import matplotlib.pyplot as plt
import quantities as pq
from scipy import signal
from scipy import interpolate
import pickle
from fractions import Fraction


def threshold_detection(signal, threshold=0.0 * pq.mV, sign='above',
                        RelaxTime=None):
    """
    Returns the times when the analog signal crosses a threshold.
    Usually used for extracting spike times from a membrane potential.
    Adapted from version in NeuroTools.

    Parameters
    ----------
    signal : neo AnalogSignal object
        'signal' is an analog signal.
    threshold : A quantity, e.g. in mV
        'threshold' contains a value that must be reached
        for an event to be detected. Default: 0.0 * mV.
    sign : 'above' or 'below'
        'sign' determines whether to count thresholding crossings
        that cross above or below the threshold.
    format : None or 'raw'
        Whether to return as SpikeTrain (None)
        or as a plain array of times ('raw').

    Returns
    -------
    result_st : neo SpikeTrain object
        'result_st' contains the spike times of each of the events (spikes)
        extracted from the signal.
    """

    assert threshold is not None, "A threshold must be provided"

    if sign == 'above':
        cutout = np.where(signal > threshold)[0]
    elif sign == 'below':
        cutout = np.where(signal < threshold)[0]

    if len(cutout) <= 0:
        events = np.zeros(0)
    else:
        take = np.where(np.diff(cutout) > 1)[0] + 1
        take = np.append(0, take)

        time = signal.times
        events = time[cutout][take]

    if RelaxTime:
        outevents = []
        told = 0*pq.s
        for te in events:
            if (te-told) > RelaxTime:
                outevents.append(te)
                told = te
    else:
        outevents = events

    return outevents


class NeoRecord():

    def __init__(self, RecordFile=None, UnitGain=1e6, Seg=None):

        self.UnitGain = UnitGain

        if not RecordFile:
            self.Seg = Seg
            if self.Seg:
                self.UpdateEventDict()
                self.UpdateSignalsDict()
            else:
                self.Seg = neo.Segment('New Seg')
            return

        ftype = RecordFile.split('.')[-1]
        if ftype == 'h5':
            self.RecFile = neo.io.NixIO(filename=RecordFile)
            Block = self.RecFile.read_block()
        elif ftype == 'smr':
            self.RecFile = neo.io.Spike2IO(filename=RecordFile)
            Block = self.RecFile.read()[0]

        self.Seg = Block.segments[0]

        self.UpdateSignalsDict()
        self.UpdateEventDict()

    def SaveRecord(self, FileName):
        out_f = neo.io.NixIO(filename=FileName)
        out_bl = neo.Block(name='NewBlock')

        out_bl.segments.append(self.Seg)
        out_f.write_block(out_bl)
        out_f.close()

    def UpdateSignalsDict(self):
        self.SigNames = {}
        for i, sig in enumerate(self.Seg.analogsignals):
            if sig.name is None:
                name = str(i)
                if sig.annotations is not None:
                    if 'nix_name' in sig.annotations.keys():
                        name = sig.annotations['nix_name']
            else:
                name = str(sig.name)

            self.SigNames.update({name: i})

    def UpdateEventDict(self):
        self.EventNames = {}
        for i, eve in enumerate(self.Seg.events):
            if eve.name is None:
                try:                    
                    name = eve.annotations['title']
                except:
                    print 'Event found no name ', i
                    name = str(i)
                self.EventNames.update({name: i})
            else:
                self.EventNames.update({eve.name: i})

    def GetEventTimes(self, EventName, Time=None):
        eve = self.Seg.events[self.EventNames[EventName]].times
        if Time:
            events = eve[np.where((eve > Time[0]) & (eve < Time[1]))]
        else:
            events = eve
        return events

    def GetTstart(self, ChName):
        return self.Seg.analogsignals[self.SigNames[ChName]].t_start

    def SetTstart(self, ChName, Tstart):
        self.Seg.analogsignals[self.SigNames[ChName]].t_start = Tstart

    def SetSignal(self, ChName, Sig):
        self.Seg.analogsignals[self.SigNames[ChName]] = Sig

    def AppendSignal(self, ChName, Vect):
        sig = self.Signal(ChName,
                          Scale=False)
        S_old = sig.copy()
        v_old = np.array(sig)
        v_new = np.vstack((v_old, Vect))
        sig_new = neo.AnalogSignal(v_new,
                                   units=S_old.units,
                                   sampling_rate=S_old.sampling_rate,
                                   t_start=S_old.t_start,
                                   name=S_old.name)
        self.SetSignal(ChName, sig_new)

    def Signal(self, ChName, Gain=None, Scale=True):
        sig = self.Seg.analogsignals[self.SigNames[ChName]]

        if Scale:
            if Gain:
                return sig/Gain
            else:
                return sig/self.UnitGain
        else:
            return sig

    def GetSignal(self, ChName, Time, Gain=None):
        sig = self.Seg.analogsignals[self.SigNames[ChName]]
        sl = sig.time_slice(Time[0], Time[1])

        if Gain:
            return sl/Gain
        else:
            return sl/self.UnitGain

    def AddEvent(self, Times, Name):
        eve = neo.Event(times=Times,
                        units=pq.s,
                        name=Name)

        self.Seg.events.append(eve)
        self.UpdateEventDict()

    def AddSignal(self, Sig):
        self.Seg.analogsignals.append(Sig)
        self.UpdateSignalsDict()

#    def AddSignal(self, ChName, Vect, Var): #  TODO check var???
#        sig = self.Signal(ChName,
#                          Scale=False)
#        S_old = sig.copy()
#        sig_new = neo.AnalogSignal(Vect,
#                                   units=S_old.units,
#                                   sampling_rate=S_old.sampling_rate,
#                                   t_start=S_old.t_start,
#                                   name=S_old.name+Var)
#        self.Seg.analogsignals.append(sig_new)
#        self.UpdateSignalsDict()


class Filter():

    FTypes = {'lp': 'lowpass',
              'hp': 'highpass',
              'bp': 'bandpass',
              'bs': 'bandstop'}

    def __init__(self, Type, Order, Freq1, Freq2=None):
        self.Type = Type
        self.Freq1 = Freq1
        self.Freq2 = Freq2
        self.Order = Order

    def ApplyFilter(self, sig):
        st = np.array(sig)
        for nf, typ in enumerate(self.Type):
            if typ == 'lp' or typ == 'hp':
                FType = self.FTypes[typ]
                Freqs = self.Freq1[nf]/(0.5*sig.sampling_rate)
            elif typ == 'bp' or typ == 'bs':
                FType = self.FTypes[typ]
                Freqs = [self.Freq1[nf]/(0.5*sig.sampling_rate),
                         self.Freq2[nf]/(0.5*sig.sampling_rate)]
            else:
                print 'Filter Type error ', typ
                continue

#            print nf, self.Order[nf], Freqs, FType
            b, a = signal.butter(self.Order[nf], Freqs, FType)
            st = signal.filtfilt(b, a, st, axis=0)

        return neo.AnalogSignal(st,
                                units=sig.units,
                                t_start=sig.t_start,
                                sampling_rate=sig.sampling_rate)


class PltSlot():
    TrtNameStr = 'T'

    def __init__(self):
        self.SpecFmin = 0.5
        self.SpecFmax = 15
        self.SpecTimeRes = 0.05
        self.SpecMinPSD = -3
        self.SpecCmap = 'jet'

        self.Position = None
        self.DispName = None
        self.SigName = None
        self.FileName = None
        self.ResamplePoints = 10000
        self.ResampleFs = None
        self.PlotType = 'Wave'
        self.Ax = None
        self.RecN = 1
        self.TStart = None #  0*pq.s
        self.AutoScale = True
        self.FiltType = ('', )
        self.FiltOrder = ('', )
        self.FiltF1 = ('', )
        self.FiltF2 = ('', )
        self.Filter = None
        self.OutType = 'I'  # 'I' 'V' 'Vg'
        self.IVGain = 1e4
        self.Gm = -1e-4
        self.GmSignal = ''
        self.rec = None

        self.LSB = None

        self.Color = 'k'
        self.Line = '-'

        self.Ymax = 0
        self.Ymin = 0

    def SetTstart(self):
        if self.TStart is not None:
            self.rec.SetTstart(self.SigName, self.TStart)

    def Resample(self, sig):
        if self.ResampleFs:
#            print 'Resamp freq', self.ResampleFs
            f = self.ResampleFs/sig.sampling_rate
            fact = Fraction(float(f)).limit_denominator()
            dowrate = fact.denominator
            uprate = fact.numerator
        else:
#            print 'Resamp points', self.ResamplePoints
            dowrate = sig.times.shape[0]/self.ResamplePoints
            if dowrate > 0:
                f = float(1/float(dowrate))
                uprate = 1

        if dowrate > 0:
            print sig.sampling_rate*f, f, uprate, dowrate
            rs = signal.resample_poly(sig, uprate, dowrate)
            sig = neo.AnalogSignal(rs,
                                   units=sig.units,
                                   t_start=sig.t_start,
                                   sampling_rate=sig.sampling_rate*f)

        if self.LSB:
            rs = np.round(sig/self.LSB)
            rs = rs * self.LSB
            sig = neo.AnalogSignal(rs,
                                   units=sig.units,
                                   t_start=sig.t_start,
                                   sampling_rate=sig.sampling_rate)

        return sig

    def Signal(self):
        return self.rec.Signal(self.SigName)

    def CheckTime(self, Time):
        if Time is None:
            return (self.Signal().t_start, self.Signal().t_stop)

        if Time[0] < self.Signal().t_start:
            Tstart = self.Signal().t_start
        else:
            Tstart = Time[0]

        if Time[1] > self.Signal().t_stop:
            Tstop = self.Signal().t_stop
        else:
            Tstop = Time[1]

        return (Tstart, Tstop)

    def ApplyGain(self, sig):
        if self.OutType == 'I':
            return sig/self.IVGain
        elif self.OutType == 'Vg':
            if self.GmSignal:
                gm = self.GmSignal[0].Signal(self.GmSignal[1])
                ti = gm.times
                gma = np.array(gm)

                if sig.t_start < gm.t_start:
                    ti = np.hstack((sig.t_start, ti))
                    gma = np.hstack((gma[0, None],
                                     gma.transpose())).transpose()

                if sig.t_stop > gm.t_stop:
                    ti = np.hstack((ti, sig.t_stop))
                    gma = np.hstack((gma.transpose(),
                                     gma[-1, None])).transpose()

                Gm = interpolate.interp1d(ti, gma, axis=0)(sig.times)
                return sig/(self.IVGain*Gm)

            return sig/(self.IVGain*self.Gm)
        else:
            return sig

    def GetSignal(self, Time, Resamp=True):
        sig = self.rec.GetSignal(self.SigName,
                                 self.CheckTime(Time))
        sig = self.ApplyGain(sig)

        if self.Filter:
            sig = self.Filter.ApplyFilter(sig)

        if Resamp:
            return self.Resample(sig)
        else:
            return sig

    def PlotSignal(self, Time, Resamp=True):
        if self.Ax:
            if self.PlotType == 'Spectrogram':
                sig = self.GetSignal(Time, Resamp=False)

                nFFT = int(2**(np.around(np.log2(sig.sampling_rate.magnitude/self.SpecFmin))+1))
                Ts = sig.sampling_period.magnitude
                noverlap = int((Ts*nFFT - self.SpecTimeRes)/Ts)

                f, t, Sxx = signal.spectrogram(sig, sig.sampling_rate,
                                               window='hanning',
                                               nperseg=nFFT,
                                               noverlap=noverlap,
                                               scaling='density',
                                               axis=0)
                finds = np.where(f < self.SpecFmax)[0][1:]
                r, g, c = Sxx.shape
                S = Sxx.reshape((r, c))[finds][:]
                self.Ax.pcolormesh(t*pq.s+sig.t_start, f[finds], np.log10(S),
                                   vmin=np.log10(np.max(S))+self.SpecMinPSD,
                                   vmax=np.log10(np.max(S)),
                                   cmap=self.SpecCmap)

            elif self.PlotType == 'Wave':
                sig = self.GetSignal(Time, Resamp)
                self.Ax.plot(sig.times, sig, self.Line, color=self.Color)

                if self.Ymin == 0 and self.Ymax == 0:
                    ylim = (np.min(sig), np.max(sig))  # TODO autoscale
                    self.Ax.set_ylim(ylim)
                else:
                    self.Ax.set_ylim((self.Ymin, self.Ymax))

    def PlotEvent(self, Time, color='r--'):
        self.Ax.plot((Time, Time), (1, -1), color, alpha=0.5)


class PlotRecord():
    FigFFT = None
    AxFFT = None

    def ClearAxes(self):
        for sl in self.Slots:
            while sl.Ax.lines:
                sl.Ax.lines[0].remove()

    def CreateFig(self, Slots):
        self.Slots = Slots

        Pos = []
        for sl in self.Slots:
            Pos.append(sl.Position)

        self.Fig, A = plt.subplots(max(Pos) + 1, 1, sharex=True)
        if type(A).__module__ == np.__name__:
            Axs = A
        else:
            Axs = []
            Axs.append(A)

        for sl in self.Slots:
            sl.SetTstart()
            setattr(sl, 'Ax', Axs[sl.Position])
            lb = sl.Ax.get_ylabel()
            sl.Ax.set_ylabel(lb + ' ' + sl.DispName)
            sl.Ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))

#            if sl.PlotType == 'Spectrogram':
#                sl.Ax.set_yscale('log')

            if not sl.FiltType[0] == '':
                print sl.FiltType, sl.FiltF1, sl.FiltF2, sl.FiltOrder
                sl.Filter = Filter(Type=sl.FiltType,
                                   Freq1=sl.FiltF1,
                                   Freq2=sl.FiltF2,
                                   Order=sl.FiltOrder)

    def PlotHist(self, Time, Resamp=False, Binds=250):
        fig, Ax = plt.subplots()

        for sl in self.Slots:
            Ax.hist(sl.GetSignal(Time, Resamp=Resamp),
                    Binds,
                    alpha=0.5)
            Ax.set_yscale('log')

    def PlotPSD(self, Time, nFFT=2**17, FMin=None, Resamp=False):

        if not self.FigFFT or not plt.fignum_exists(self.FigFFT.number):
            self.FigFFT, self.AxFFT = plt.subplots()

        for sl in self.Slots:
            sig = sl.GetSignal(Time, Resamp=Resamp)
            if FMin:
                nFFT = int(2**(np.around(np.log2(sig.sampling_rate.magnitude/FMin))+1)) 

            ff, psd = signal.welch(x=sig, fs=sig.sampling_rate,
                                   window='hanning',
                                   nperseg=nFFT, scaling='density', axis=0)

            self.AxFFT.loglog(ff, psd, label=sl.DispName)

        self.AxFFT.set_xlabel('Frequency [Hz]')
        self.AxFFT.set_ylabel('PSD [V^2/Hz]')
        self.AxFFT.legend()

    def PlotEventAvg(self, (EventRec, EventName), Time, TimeWindow,
                     OverLap=True, Std=False, Spect=False, Resamp=False):

        ft, Axt = plt.subplots()

        for sl in self.Slots:
            avg = np.array([])
            if Spect:
                ft, (Ax, AxS) = plt.subplots(2, 1, sharex=True)
            else:
                ft, Ax = plt.subplots()

            if Resamp:
                Fs = sl.ResampleFs.magnitude
            else:
                Fs = sl.Signal().sampling_rate.magnitude

            Ts = 1/Fs
            nSamps = int((TimeWindow[1]-TimeWindow[0])/Ts)
            t = np.arange(nSamps)*Ts*pq.s + TimeWindow[0]

            etimes = EventRec.GetEventTimes(EventName, Time)
            for et in etimes:
                start = et+TimeWindow[0]
                stop = et+TimeWindow[1]

                if sl.Signal().t_start < start and sl.Signal().t_stop > stop:
                    st = sl.GetSignal((start, stop), Resamp=Resamp)[:nSamps]
                    try:
                        avg = np.hstack([avg, st]) if avg.size else st
                        if OverLap:
                            Ax.plot(t, st, 'k-', alpha=0.1)
                    except:
                        print 'Error', nSamps, et, avg.shape, st.shape

            print EventName, 'Counts', len(etimes)

            MeanT = np.mean(avg, axis=1)
            Ax.plot(t, MeanT, 'r-')
            if Std:
                StdT = np.std(avg, axis=1)
                Ax.fill_between(t, MeanT+StdT, MeanT-StdT,
                                facecolor='r', alpha=0.5)

            ylim = Ax.get_ylim()
            Ax.plot((0, 0), (1, -1), 'g-.', alpha=0.7)
            Ax.set_ylim(ylim)
            Ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))

            plt.title(sl.DispName)
            if Spect:
                nFFT = int(2**(np.around(np.log2(Fs/sl.SpecFmin))+1))
                noverlap = int((Ts*nFFT - sl.SpecTimeRes)/Ts)
                TWindOff = (nFFT * Ts)/8

                f, tsp, Sxx = signal.spectrogram(MeanT, Fs,
                                                 window='hanning',
                                                 nperseg=nFFT,
                                                 noverlap=noverlap,
                                                 scaling='density',
                                                 axis=0)

                finds = np.where(f < sl.SpecFmax)[0][1:]
                print Sxx.shape
                r, c = Sxx.shape
                S = Sxx.reshape((r, c))[finds][:]
                pcol = AxS.pcolormesh(tsp + TimeWindow[0].magnitude + TWindOff,
                                      f[finds],
                                      np.log10(S),
                                      vmin=np.log10(np.max(S))+sl.SpecMinPSD,
                                      vmax=np.log10(np.max(S)),
                                      cmap=sl.SpecCmap)
                f, a = plt.subplots(1, 1)
                f.colorbar(pcol)

            Axt.plot(t, np.mean(avg, axis=1), label=sl.DispName)
            ft.canvas.draw()

        Axt.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
        Axt.legend()
        ft.canvas.draw()
        plt.show()

    def PlotPSDSNR(self, (EventRec, EventName), TDelay, TEval, DevDCVals,
                   Time=None, nFFT=2**17):

        etimes = EventRec.GetEventTimes(EventName, Time)

        DevACVals = {}

        for sl in self.Slots:
            fig, (ax1, ax2)  = plt.subplots(1,2)
            psd = np.array([])
            DCVals = DevDCVals[sl.SigName]
            for ne, et in enumerate(etimes):
                start = et+TDelay
                stop = et+TDelay+TEval

                sig = sl.GetSignal((start, stop), Resamp=False)
                fpsd, npsd = signal.welch(x=sig, fs=sig.sampling_rate,
                                          window='hanning',
                                          nperseg=nFFT, scaling='density', axis=0)                
                psd = np.hstack([psd, npsd]) if psd.size else npsd

#                Flin = fpsd[1:].magnitude
#                Flog = np.logspace(np.log10(Flin[0]),
#                                   np.log10(Flin[-1]), 100)
#                Flog = np.round(Flog, 9)
#                Flin = np.round(Flin, 9)
#                intpsd = interpolate.interp1d(Flin, npsd[1:].transpose())(Flog)
#                a, b, _ = FETAna.noise.FitNoise(Flog,
#                                                intpsd, Fmin=150, Fmax=5e3)

                ax1.loglog(fpsd, npsd)
#                ax2.loglog(Flog, intpsd/FETAna.noise(Flog, a, b))

            psdD = {'Vd0': psd.transpose()}
            ACVals = {'PSD': psdD,
                      'gm': None,
                      'Vgs': DCVals['Vgs'],
                      'Vds': DCVals['Vds'],
                      'Fpsd': fpsd.magnitude,
                      'Fgm': None,
                      'ChName': sl.SigName,
                      'Name': sl.DispName,
                      'GMPoly': DCVals['GMPoly'],
                      'IdsPoly': DCVals['IdsPoly'],
                      'Ud0': DCVals['Ud0'],
                      'IsOK': DCVals['IsOK'],
                      'DateTime': DCVals['DateTime']}
            DevACVals[sl.DispName] = ACVals
            fig.canvas.draw()
            plt.show()

        FETAna.InterpolatePSD(DevACVals)
        FETAna.FitACNoise(DevACVals, Fmin=150, Fmax=5e3, IsOkFilt=False)
        pltPSD = FETplt.PyFETPlot()
        pltPSD.AddAxes(('PSD', 'NoA', 'NoB'))
        pltPSD.AddLegend()
        pltPSD.PlotDataCh(DevACVals)

        pickle.dump(DevACVals, open('test.pkl', 'wb'))

        return DevACVals

    def PlotChannels(self, Time, Resamp=True,
                     ResampPoints=None, ResampFs=None):

        if not self.Fig:
            return

        for sl in self.Slots:
            if ResampPoints:
                sl.ResamplePoints = ResampPoints
            if ResampFs:
                sl.ResampFs = ResampFs

            sl.PlotSignal(Time, Resamp=Resamp)

        sl.Ax.set_xlim(Time[0], Time[1])

    def PlotEvents(self, Time, (EventRec, EventName)):

        Te = EventRec.GetEventTimes(EventName, Time)
        for te in Te:
            for sl in self.Slots:
                sl.PlotEvent(te)

