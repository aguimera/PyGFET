# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 12:57:20 2017

@author: eduard
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import gridspec
from PyGFET.RecordCore import NeoRecord
from PyGFET.RecordPlot import PltSlot, PlotRecord
import quantities as pq
import matplotlib.colors as mpcolors

plt.close('all')

RecordFile = 'TestData.h5'

RecDC = NeoRecord(RecordFile, UnitGain=1)

Slots = []
for i, Sname in enumerate(sorted(RecDC.SigNames.keys())):
    sl = PltSlot()
    sl.rec = RecDC
    sl.Position = i
    sl.SigName = Sname
    sl.DispName = Sname
    sl.OutType = 'V'
#    sl.Ymax = 0.5e-3
#    sl.Ymin = -0.5e-3
    Slots.append(sl)    

for i, Sname in enumerate(sorted(RecDC.SigNames.keys())):
    sl = PltSlot()
    sl.FiltType = ('lp', )
    sl.FiltOrder = (3, )
    sl.FiltF1 = (1, )
    sl.Color = 'b'
    sl.rec = RecDC
    sl.Position = i
    sl.SigName = Sname
    sl.DispName = Sname + 'F'
    sl.OutType = 'V'
    Slots.append(sl)
    
for i, Sname in enumerate(sorted(RecDC.SigNames.keys())):
    sl = PltSlot()
    sl.FiltType = ('hp', )
    sl.FiltOrder = (3, )
    sl.FiltF1 = (1, )
    sl.Color = 'r'
    sl.rec = RecDC
    sl.Position = i
    sl.SigName = Sname
    sl.DispName = Sname + 'F2'
    sl.OutType = 'V'
    Slots.append(sl)

PltRecs = PlotRecord()
PltRecs.LegNlabCol = 1
PltRecs.CreateFig(Slots, ShowLegend=True)

Tstart = 0*pq.s
Tstop = 50*pq.s
TShow = (Tstart, Tstop)

PltRecs.ClearAxes()
PltRecs.PlotChannels(TShow, Resamp=False)


#Reltime=0
#def f(Reltime,PltRecs):
#    Z=np.ones((4,4))
#    for sl in PltRecs.Slots:
#        sig = sl.GetSignal(Twind, Resamp=False)
#        Z[(ChXYlocation[sl.DispName])]=sig[Reltime]
#        
#    return Z        
#
#
##==============================================================================
## W=np.ones((4,4))
## for sl in PltRecs.Slots:
#     sig = sl.GetSignal(Twind, Resamp=False)
#     W[(ChXYlocation[sl.DispName])]=sig[0]
#==============================================================================

#==============================================================================
# cmap = cmx.ScalarMappable(mpcolors.Normalize(vmin=-20e-3, vmax=0),cmx.copper)
#==============================================================================
# def data_gen():
#     cnt = 0
#     while cnt < (Tstop-Tstart)*2:
#         cnt += 1
#         yield cnt, f(cnt)
#==============================================================================
#
#fig = plt.figure()
#gs = gridspec.GridSpec(2, 1,                          
#                       height_ratios=[1, 8])
#Ax1= fig.add_subplot(gs[0])
#
#Ax1.set_xlabel('Time [s]')
##Ax1.get_yaxis().set_visible(False)
#Ax1.spines['top'].set_visible(False)
#Ax1.spines['right'].set_visible(False)
#Ax1.spines['left'].set_visible(False)
#Ax1.spines['bottom'].set_visible(False)
#Ax1.tick_params(axis='both', which='major', labelsize='small')
#
#Ax2= fig.add_subplot(gs[1])
#Ax2.set_xticks(range(4))
#Ax2.set_yticks(range(4))
#Ax2.set_xticklabels(np.arange(4)*0.4)
#Ax2.set_yticklabels(np.arange(4)*0.4)
#Ax2.set_xlabel('X [mm]')
#Ax2.set_ylabel('Y [mm]')
#Ax2.spines['top'].set_visible(False)
#Ax2.spines['right'].set_visible(False)
#Ax2.spines['left'].set_visible(False)
#Ax2.spines['bottom'].set_visible(False)
#
#
#fig.tight_layout()
#
##==============================================================================
## im = plt.imshow(f(frames,PltRecs),cmap='viridis',norm=mpcolors.Normalize(vmin=-25e-3, vmax=5e-3),animated=True)
##==============================================================================
##==============================================================================
## im = plt.imshow(f(Reltime,PltRecs),cmap='viridis', norm=mpcolors.Normalize(vmin=-25e-3, vmax=5e-3),animated=True)
##==============================================================================
#im = Ax2.imshow(f(Reltime,PltRecs),
#                cmap='seismic',
#                norm=mpcolors.Normalize(vmin=-30e-3, vmax=30e-3),
#                animated=True,
#                interpolation='bicubic')
#
#
#cbar = plt.colorbar(im)
#cbar.ax.tick_params(labelsize='x-small') 
#cbar.set_label('[V]')
#
#slw = PltSlot()
#slw.rec = RecDC
#slw.SigName = 'Ch08'
#slw.DispName = 'Ch08'
#slw.OutType = 'V'
#
#sigw = slw.GetSignal(Twind, Resamp=False)
#
#Emax = np.max(sigw)
#Emin = np.min(sigw)
#times = sigw.times
#times = times - times[0]
#wav, = Ax1.plot(times,sigw,'k-')
#wavt, = Ax1.plot([0,0],(Emax,Emin),'r--')
#
#dy,dx=np.gradient(f(Reltime,PltRecs))         
#Q=Ax2.quiver(dx, dy,scale=0.2)
#                       
#def updatefig(*args):
#    global Reltime, PltRecs
#    Reltime+=1
#    im.set_array(f(Reltime,PltRecs))
#    wavt.set_data([Reltime,Reltime],[Emax,Emin])
#    dy,dx=np.gradient(f(Reltime,PltRecs))
#    Q.set_UVC(dx,dy)
#    return im, wavt, Q
#
#ani = animation.FuncAnimation(fig, updatefig, frames=TotalTime, interval=200, blit=False)
#plt.show()
#
#
### Set up formatting for the movie files
##Writer = animation.writers['ffmpeg']
##writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)
##ani.save('Rec4Spreading_5x_interp.mp4',writer=writer)
#
#
##ani = animation.FuncAnimation(fig, updatefig, frames=TotalTime, interval=1, blit=True)
##plt.show()
##
##Writer = animation.writers['ffmpeg']
##writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)
##ani.save('Rec4Spreading_5x_interp_null.mp4',writer=writer,savefig_kwargs={'fname':'ft.png'})
#
#
#

