#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 11:39:44 2017

@author: aguimera
"""

import neo
import gc
import quantities as pq
import glob


FileFilter = '../../171010/Rec9/B10179W11B8_d171010_rec9_SD.smr'

FileNames = glob.glob(FileFilter)

for ifile, filen in enumerate(FileNames):

    print ifile, filen    
    
    reader = neo.io.Spike2IO(filename=filen)
    in_bl = reader.read()[0]
    in_seg = in_bl.segments[0]
    
    inSigNames={}
    for i, sig in enumerate(in_seg.analogsignals):
        inSigNames.update({sig.name:i})        
    
    out_f = neo.io.NixIO(filename=filen.replace('smr','h5'))
    
    out_seg = neo.Segment(name='NewSeg')
    
    for k,v in inSigNames.iteritems():
#        if not k.startswith('T'):continue
        print 'Signal Found', sig.name
        sig = in_seg.analogsignals[v]*pq.V
        out_seg.analogsignals.append(sig)
    
    for e in in_seg.events:
        print 'Event Found', e.annotations['title']
        e.name = e.annotations['title']
        out_seg.events.append(e)
        
    out_bl = neo.Block(name='NewBlock')
    
    out_bl.segments.append(out_seg)
    
    out_f.write_block(out_bl)
    
    out_f.close()

print 'Collect ', gc.collect()

