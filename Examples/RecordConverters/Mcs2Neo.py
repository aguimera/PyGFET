#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 22:31:57 2017

@author: aguimera
"""
import matplotlib.pyplot as plt
import McsPy.McsData as McsData
import quantities as pq
import neo
import numpy as np

plt.close('all')


FileName = 'S:\\Stimulation\\In vivo\\EGNITE Paris\\29092017\\Exported data\\pedot_retina1b_polychrome_450nm_2sPreblock0001.h5'
FileOut =  'S:\\Stimulation\\In vivo\\EGNITE Paris\\29092017\\Exported data\\NeoFiles\\pedot_retina1b_polychrome_450nm_2sPreblock0001_v2.h5'

out_f = neo.io.NixIO(filename=FileOut)   
out_seg = neo.Segment(name='NewSeg')
out_bl = neo.Block(name='NewBlock')

Dat = McsData.RawData(FileName)
Rec = Dat.recordings[0]

NSamps = Rec.duration


for AnaStrn, AnaStr in Rec.analog_streams.iteritems():
    for Chn, Chinfo in AnaStr.channel_infos.iteritems():        
        print 'Analog Stream ', Chinfo.label, Chinfo.sampling_frequency
        print Chinfo.info['Unit']
        Chinfo.info['Unit']='V'
        
        Fs = Chinfo.sampling_frequency
        
        Var,Unit = AnaStr.get_channel_in_range(Chn,0,NSamps)
        
        sig = neo.AnalogSignal(np.array(Var),
                               units = pq.V,
                               t_start = 0*pq.s,
                               sampling_rate = Fs.magnitude*pq.Hz,
                               name = Chinfo.label)
        
        out_seg.analogsignals.append(sig) 


#==============================================================================
# for EventN, Event in Rec.event_streams.iteritems():
#     print Event.label
#     Times, Unit = Event.event_entity[0].get_event_timestamps()
#     
#     eve = neo.Event(times=Times/1e6,
#                     units=pq.s,
#                     name=Event.label)
# 
#     out_seg.events.append(eve)
#==============================================================================
    
out_bl.segments.append(out_seg)
out_f.write_block(out_bl)
out_f.close()

#
#for Chn,Vals in DCData.iteritems():
#    if Chn.startswith('Ch'):
#        for Varn,Var in Vals.iteritems():
#            if Varn=='tperiod' or Varn=='tstart' or Varn=='Vds' or Varn=='Vgs': continue
#            
#            sig = neo.AnalogSignal(Var,
#                                       units = Units[Varn],
#                                       t_start = Vals['tstart']*pq.s,
#                                       sampling_rate = (1/Vals['tperiod'])*pq.Hz,
#                                       name=Chn+Varn)
#            out_seg.analogsignals.append(sig)            
#    else:
#        sig = neo.AnalogSignal(Vals,
#                               units = pq.V,
#                               t_start = 0*pq.s,
#                               sampling_rate = 1*pq.Hz,
#                               name=Chn)
#        out_seg.analogsignals.append(sig) 





#    inf = Ana.channel_infos[Chinfo]
#    inf.info['Unit']=''
#    idc = inf.channel_id
#    
#    Vals = Ana.get_channel_in_range(idc,0,nSamps/1000)
#    Times = Ana.get_channel_sample_timestamps(idc,0,nSamps/1000)
#    plt.plot(Times[0],Vals[0])









