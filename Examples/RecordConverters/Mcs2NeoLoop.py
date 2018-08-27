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
import glob

plt.close('all')

FileFilter = 'S:\\Stimulation\\In vivo\\EGNITE Paris\\29092017\\Exported data\\egnite_retina2_polychrome_450nm_2s0001.h5'
FileNames = glob.glob(FileFilter)

for FileName in FileNames:
    print FileName
    FileOut = FileName.split('.')[0]+'_v2.h5'
    print FileOut

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
                                   name = Chinfo.label,
                                   file_origin=FileName)
            
            out_seg.analogsignals.append(sig) 
            
        if Rec.event_streams is None:
            continue

        for EventN, Event in Rec.event_streams.iteritems():
            print Event.label
            Times, Unit = Event.event_entity[0].get_event_timestamps()
            
            eve = neo.Event(times=Times/1e6,
                             units=pq.s,
                             name=Event.label)
         
            out_seg.events.append(eve)
   
    out_bl.segments.append(out_seg)
    out_f.write_block(out_bl)
    out_f.close()






