#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 14:44:29 2017

@author: aguimera
"""

#### 64CH MCS channel Names

MCS64Col1 = {'Ch03': 'Ch49',
             'Ch07': 'Ch50',
             'Ch04': 'Ch51',
             'Ch08': 'Ch52',
             'Ch14': 'Ch53',
             'Ch10': 'Ch54',
             'Ch13': 'Ch55',
             'Ch09': 'Ch56',
             'Ch01': 'Ch57',
             'Ch05': 'Ch58',
             'Ch02': 'Ch59',
             'Ch06': None,
             'Ch16': None,
             'Ch12': None,
             'Ch15': None,
             'Ch11': None}

MCS64Col2 = {'Ch03': 'Ch47',
             'Ch07': 'Ch46',
             'Ch04': 'Ch45',
             'Ch08': 'Ch44',
             'Ch14': 'Ch43',
             'Ch10': 'Ch65',
             'Ch13': 'Ch64',
             'Ch09': 'Ch63',
             'Ch01': 'Ch62',
             'Ch05': 'Ch61',
             'Ch02': 'Ch60',
             'Ch06': None,
             'Ch16': None,
             'Ch12': None,
             'Ch15': None,
             'Ch11': None}

MCS64Col3 = {'Ch03': 'Ch37',
             'Ch07': 'Ch38',
             'Ch04': 'Ch39',
             'Ch08': 'Ch40',
             'Ch14': 'Ch41',
             'Ch10': 'Ch42',
             'Ch13': 'Ch66',
             'Ch09': 'Ch67',
             'Ch01': 'Ch68',
             'Ch05': 'Ch69',
             'Ch02': 'Ch70',
             'Ch06': None,
             'Ch16': None,
             'Ch12': None,
             'Ch15': None,
             'Ch11': None}

MCS64Col4 = {'Ch03': 'Ch34',
             'Ch07': 'Ch33',
             'Ch04': 'Ch32',
             'Ch08': 'Ch31',
             'Ch14': 'Ch30',
             'Ch10': 'Ch06',
             'Ch13': 'Ch05',
             'Ch09': 'Ch04',
             'Ch01': 'Ch03',
             'Ch05': 'Ch02',
             'Ch02': 'Ch01',
             'Ch06': None,
             'Ch16': None,
             'Ch12': None,
             'Ch15': None,
             'Ch11': None}

MCS64Col5 = {'Ch03': 'Ch24',
             'Ch07': 'Ch25',
             'Ch04': 'Ch26',
             'Ch08': 'Ch27',
             'Ch14': 'Ch28',
             'Ch10': 'Ch29',
             'Ch13': 'Ch07',
             'Ch09': 'Ch08',
             'Ch01': 'Ch09',
             'Ch05': 'Ch10',
             'Ch02': 'Ch11',
             'Ch06': None,
             'Ch16': None,
             'Ch12': None,
             'Ch15': None,
             'Ch11': None}

MCS64Col6 = {'Ch03': 'Ch22',
             'Ch07': 'Ch21',
             'Ch04': 'Ch20',
             'Ch08': 'Ch19',
             'Ch14': 'Ch18',
             'Ch10': 'Ch17',
             'Ch13': 'Ch16',
             'Ch09': 'Ch15',
             'Ch01': 'Ch14',
             'Ch05': 'Ch13',
             'Ch02': 'Ch12',
             'Ch06': None,
             'Ch16': None,
             'Ch12': None,
             'Ch15': None,
             'Ch11': None}

MCS64ColMAP = {'Col1': MCS64Col1,
               'Col2': MCS64Col2,
               'Col3': MCS64Col3,
               'Col4': MCS64Col4,
               'Col5': MCS64Col5,
               'Col6': MCS64Col6}


### Flat contact new N W=5um
RW50L50P3Cov10 = {'Name': 'RW50L50P3Cov20',
                  'Length': 50e-6,
                  'Width': 50e-6,
                  'Area': 2.5e-09,
                  'Pass': 3e-6,
                  'Contact': 'Flat10um',
                  'Shape': 'Rectangular'}

RW20L20P3Cov10 = {'Name': 'RW20L20P3Cov10',
                  'Length': 20e-6,
                  'Width': 20e-6,
                  'Area': 4e-10,
                  'Pass': 3e-6,
                  'Contact': 'Flat10um',
                  'Shape': 'Rectangular'}

MCS64TrtTypes = {'Ch01': RW50L50P3Cov10,
                 'Ch02': RW20L20P3Cov10,
                 'Ch03': RW50L50P3Cov10,
                 'Ch04': RW20L20P3Cov10,
                 'Ch05': RW50L50P3Cov10,
                 'Ch06': RW20L20P3Cov10,
                 'Ch07': RW20L20P3Cov10,
                 'Ch08': RW50L50P3Cov10,
                 'Ch09': RW20L20P3Cov10,
                 'Ch10': RW50L50P3Cov10,
                 'Ch11': RW20L20P3Cov10,
                 'Ch12': RW50L50P3Cov10,
                 'Ch13': RW20L20P3Cov10,
                 'Ch14': RW50L50P3Cov10,
                 'Ch15': RW20L20P3Cov10,
                 'Ch16': RW50L50P3Cov10,
                 'Ch17': RW20L20P3Cov10,
                 'Ch18': RW50L50P3Cov10,
                 'Ch19': RW20L20P3Cov10,
                 'Ch20': RW50L50P3Cov10,
                 'Ch21': RW20L20P3Cov10,
                 'Ch22': RW50L50P3Cov10,
                 'Ch23': None,
                 'Ch24': RW20L20P3Cov10,
                 'Ch25': RW50L50P3Cov10,
                 'Ch26': RW20L20P3Cov10,
                 'Ch27': RW50L50P3Cov10,
                 'Ch28': RW20L20P3Cov10,
                 'Ch29': RW50L50P3Cov10,
                 'Ch30': RW50L50P3Cov10,
                 'Ch31': RW20L20P3Cov10,
                 'Ch32': RW50L50P3Cov10,
                 'Ch33': RW20L20P3Cov10,
                 'Ch34': RW50L50P3Cov10,
                 'Ch35': None,
                 'Ch36': None,
                 'Ch37': RW20L20P3Cov10,
                 'Ch38': RW50L50P3Cov10,
                 'Ch39': RW20L20P3Cov10,
                 'Ch40': RW50L50P3Cov10,
                 'Ch41': RW20L20P3Cov10,
                 'Ch42': RW50L50P3Cov10,
                 'Ch43': RW50L50P3Cov10,
                 'Ch44': RW20L20P3Cov10,
                 'Ch45': RW50L50P3Cov10,
                 'Ch46': RW20L20P3Cov10,
                 'Ch47': RW50L50P3Cov10,
                 'Ch48': None,
                 'Ch49': RW20L20P3Cov10,
                 'Ch50': RW50L50P3Cov10,
                 'Ch51': RW20L20P3Cov10,
                 'Ch52': RW50L50P3Cov10,
                 'Ch53': RW20L20P3Cov10,
                 'Ch54': RW50L50P3Cov10,
                 'Ch55': RW20L20P3Cov10,
                 'Ch56': RW50L50P3Cov10,
                 'Ch57': RW20L20P3Cov10,
                 'Ch58': RW50L50P3Cov10,
                 'Ch59': RW20L20P3Cov10,
                 'Ch60': RW50L50P3Cov10,
                 'Ch61': RW20L20P3Cov10,
                 'Ch62': RW50L50P3Cov10,
                 'Ch63': RW20L20P3Cov10,
                 'Ch64': RW50L50P3Cov10,
                 'Ch65': RW20L20P3Cov10,
                 'Ch66': RW20L20P3Cov10,
                 'Ch67': RW50L50P3Cov10,
                 'Ch68': RW20L20P3Cov10,
                 'Ch69': RW50L50P3Cov10,
                 'Ch70': RW20L20P3Cov10}


#### MUX channel names

TrtLine1 = {'Ch02': 'Ch11',
            'Ch03': 'Ch12',
            'Ch04': 'Ch13',
            'Ch05': 'Ch14',
            'Ch06': 'Ch15',
            'Ch07': 'Ch16',
            'Ch08': 'Ch17',
            'Ch09': 'Ch18'}

TrtLine2 = {'Ch02': 'Ch21',
            'Ch03': 'Ch22',
            'Ch04': 'Ch23',
            'Ch05': 'Ch24',
            'Ch06': 'Ch25',
            'Ch07': 'Ch26',
            'Ch08': 'Ch27',
            'Ch09': 'Ch28'}

TrtLine3 = {'Ch02': 'Ch31',
            'Ch03': 'Ch32',
            'Ch04': 'Ch33',
            'Ch05': 'Ch34',
            'Ch06': 'Ch35',
            'Ch07': 'Ch36',
            'Ch08': 'Ch37',
            'Ch09': 'Ch38'}

TrtLine4 = {'Ch02': 'Ch41',
            'Ch03': 'Ch42',
            'Ch04': 'Ch43',
            'Ch05': 'Ch44',
            'Ch06': 'Ch45',
            'Ch07': 'Ch46',
            'Ch08': 'Ch47',
            'Ch09': 'Ch48'}

TrtLine5 = {'Ch02': 'Ch51',
            'Ch03': 'Ch52',
            'Ch04': 'Ch53',
            'Ch05': 'Ch54',
            'Ch06': 'Ch55',
            'Ch07': 'Ch56',
            'Ch08': 'Ch57',
            'Ch09': 'Ch58'}

TrtLine6 = {'Ch02': 'Ch61',
            'Ch03': 'Ch62',
            'Ch04': 'Ch63',
            'Ch05': 'Ch64',
            'Ch06': 'Ch65',
            'Ch07': 'Ch66',
            'Ch08': 'Ch67',
            'Ch09': 'Ch68'}

TrtLine7 = {'Ch02': 'Ch71',
            'Ch03': 'Ch72',
            'Ch04': 'Ch73',
            'Ch05': 'Ch74',
            'Ch06': 'Ch75',
            'Ch07': 'Ch76',
            'Ch08': 'Ch77',
            'Ch09': 'Ch78'}

TrtLine8 = {'Ch02': 'Ch81',
            'Ch03': 'Ch82',
            'Ch04': 'Ch83',
            'Ch05': 'Ch84',
            'Ch06': 'Ch85',
            'Ch07': 'Ch86',
            'Ch08': 'Ch87',
            'Ch09': 'Ch88'}



#### DUT channel names

TrtName1=  {'Ch02':'T3','Ch03':None, 'Ch04':'T4', 'Ch05':'T1', 'Ch06':'T5', 'Ch07':'T2', 'Ch08':'T6','Ch09':None,
                 'Ch10':'T10','Ch11':None, 'Ch12':'T9', 'Ch13':None, 'Ch14':'T8', 'Ch15':'T11', 'Ch16':'T7',
                 'Gate':'Gate'}   # TrtName= TipName-1;           

TrtName13 = {'Ch02':'T15','Ch03':None, 'Ch04':'T14', 'Ch05':'T13', 'Ch06':'T17', 'Ch07':'T12', 'Ch08':'T16','Ch09':None,
                 'Ch10':'T20','Ch11':None, 'Ch12':'T21', 'Ch13':'T22', 'Ch14':'T18', 'Ch15':None, 'Ch16':'T19',
                 'Gate':'Gate'}
    
### Flat contacts N

RW40L40P1p5CS = {'Name':'RW40L40P1p5CS',
                 'Length':40e-6, 
                 'Width':40e-6,
                 'Area':1600e-12,
                 'Pass':1.5e-6,
                 'Contact':'Flat',
                 'Shape':'Rectangular'}                 

RW40L20P1p5CS = {'Name':'RW40L20P1p5CS',
                 'Length':20e-6, 
                 'Width':40e-6,
                 'Area':800e-12,
                 'Pass':1.5e-6,
                 'Contact':'Flat',
                 'Shape':'Rectangular'}                 
                 
RW40L10P1p5CS = {'Name':'RW40L10P1p5CS',
                 'Length':10e-6, 
                 'Width':40e-6,
                 'Area':400e-12,
                 'Pass':1.5e-6,
                 'Contact':'Flat',
                 'Shape':'Rectangular'}                 

RW40L5P1p5CS = {'Name':'RW40L5P1p5CS',
                 'Length':5e-6, 
                 'Width':40e-6,
                 'Area':200e-12,
                 'Pass':1.5e-6,
                 'Contact':'Flat',
                 'Shape':'Rectangular'}                 

RW40L2p5P1p5CS = {'Name':'RW40L2p5P1p5CS',
                 'Length':2.5e-6, 
                 'Width':40e-6,
                 'Area':100e-12,
                 'Pass':1.5e-6,
                 'Contact':'Flat2um',
                 'Shape':'Rectangular'}  
           

### Circles 5um contacts S

RW40L40P1p5CC5 = {'Name':'RW40L40P1p5CC5',
                 'Length':40e-6, 
                 'Width':40e-6,
                 'Area':1600e-12,
                 'Pass':1.5e-6,
                 'Contact':'Circles 5',
                 'Shape':'Rectangular'}                 

RW40L20P1p5CC5 = {'Name':'RW40L20P1p5CC5',
                 'Length':20e-6, 
                 'Width':40e-6,
                 'Area':800e-12,
                 'Pass':1.5e-6,
                 'Contact':'Circles 5',
                 'Shape':'Rectangular'}                 

RW40L10P1p5CC5 = {'Name':'RW40L10P1p5CC5',
                 'Length':10e-6, 
                 'Width':40e-6,
                 'Area':400e-12,
                 'Pass':1.5e-6,
                 'Contact':'Circles 5',
                 'Shape':'Rectangular'}                 

RW40L5P1p5CC5 = {'Name':'RW40L5P1p5CC5',
                 'Length':5e-6, 
                 'Width':40e-6,
                 'Area':200e-12,
                 'Pass':1.5e-6,
                 'Contact':'Circles 5',
                 'Shape':'Rectangular'}                 

RW40L2p5P1p5CC5 = {'Name':'RW40L2p5P1p5CC5',
                 'Length':2.5e-6, 
                 'Width':40e-6,
                 'Area':100e-12,
                 'Pass':1.5e-6,
                 'Contact':'Circles 5',
                 'Shape':'Rectangular'}          

### Fingers 2um contacts E

RW40L40P1p5CF2 = {'Name':'RW40L40P1p5CF2',
                 'Length':40e-6, 
                 'Width':40e-6,
                 'Area':1600e-12,
                 'Pass':1.5e-6,
                 'Contact':'Fingers 2',
                 'Shape':'Rectangular'}                 

RW40L20P1p5CF2 = {'Name':'RW40L20P1p5CF2',
                 'Length':20e-6, 
                 'Width':40e-6,
                 'Area':800e-12,
                 'Pass':1.5e-6,
                 'Contact':'Fingers 2',
                 'Shape':'Rectangular'}                 

RW40L10P1p5CF2 = {'Name':'RW40L10P1p5CF2',
                 'Length':10e-6, 
                 'Width':40e-6,
                 'Area':400e-12,
                 'Pass':1.5e-6,
                 'Contact':'Fingers 2',
                 'Shape':'Rectangular'}                 

RW40L5P1p5CF2 = {'Name':'RW40L5P1p5CF2',
                 'Length':5e-6, 
                 'Width':40e-6,
                 'Area':200e-12,
                 'Pass':1.5e-6,
                 'Contact':'Fingers 2',
                 'Shape':'Rectangular'}                 

RW40L2p5P1p5CF2 = {'Name':'RW40L2p5P1p5CF2',
                 'Length':2.5e-6, 
                 'Width':40e-6,
                 'Area':100e-12,
                 'Pass':1.5e-6,
                 'Contact':'Fingers 2',
                 'Shape':'Rectangular'} 


### Fingers 3.5um contacts O

RW40L40P1p5CF3p5 = {'Name':'RW40L40P1p5CF3p5',
                 'Length':40e-6, 
                 'Width':40e-6,
                 'Area':1600e-12,
                 'Pass':1.5e-6,
                 'Contact':'Fingers 3.5',
                 'Shape':'Rectangular'}                 

RW40L20P1p5CF3p5 = {'Name':'RW40L20P1p5CF3p5',
                 'Length':20e-6, 
                 'Width':40e-6,
                 'Area':800e-12,
                 'Pass':1.5e-6,
                 'Contact':'Fingers 3.5',
                 'Shape':'Rectangular'}                 

RW40L10P1p5CF3p5 = {'Name':'RW40L10P1p5CF3p5',
                 'Length':10e-6, 
                 'Width':40e-6,
                 'Area':400e-12,
                 'Pass':1.5e-6,
                 'Contact':'Fingers 3.5',
                 'Shape':'Rectangular'}                 

RW40L5P1p5CF3p5 = {'Name':'RW40L5P1p5CF3p5',
                 'Length':5e-6, 
                 'Width':40e-6,
                 'Area':200e-12,
                 'Pass':1.5e-6,
                 'Contact':'Fingers 3.5',
                 'Shape':'Rectangular'}                 

RW40L2p5P1p5CF3p5 = {'Name':'RW40L2p5P1p5CF3p5',
                 'Length':2.5e-6, 
                 'Width':40e-6,
                 'Area':100e-12,
                 'Pass':1.5e-6,
                 'Contact':'Fingers 3.5',
                 'Shape':'Rectangular'}     
                 
                 

RW40L15P0p5CS = {'Name':'RW40L15P0p5CS',
                 'Length':15e-6, 
                 'Width':40e-6,
                 'Area':600e-12,
                 'Pass':0.5e-6,
                 'Contact':'Flat',
                 'Shape':'Rectangular'}  

RW40L15P1p0CS = {'Name':'RW40L15P1p0CS',
                 'Length':15e-6, 
                 'Width':40e-6,
                 'Area':600e-12,
                 'Pass':1.0e-6,
                 'Contact':'Flat',
                 'Shape':'Rectangular'}  
                 
RW40L15P2p0CS = {'Name':'RW40L15P2p0CS',
                 'Length':15e-6, 
                 'Width':40e-6,
                 'Area':600e-12,
                 'Pass':2.0e-6,
                 'Contact':'Flat',
                 'Shape':'Rectangular'}  
                 
RW40L15P2p0CSOv = {'Name':'RW40L15P2p0CSOv',
                 'Length':15e-6, 
                 'Width':40e-6,
                 'Area':600e-12,
                 'Pass':2.0e-6,
                 'Contact':'PassOverlap',
                 'Shape':'Rectangular'}  

#circular/rectangulars N
RW80L5P5p0CS = {'Name':'RW80L5P5p0CS',
                 'Length':5e-6, 
                 'Width':80e-6,
                 'Area':400e-12,
                 'Pass':5.0e-6,
                 'Contact':'Flat',
                 'Shape':'Rectangular'}
RW80L10P5p0CS = {'Name':'RW80L10P5p0CS',
                 'Length':10e-6, 
                 'Width':80e-6,
                 'Area':800e-12,
                 'Pass':5.0e-6,
                 'Contact':'Flat',
                 'Shape':'Rectangular'}
RW80L20P5p0CS = {'Name':'RW80L20P5p0CS',
                 'Length':20e-6, 
                 'Width':80e-6,
                 'Area':1600e-12,
                 'Pass':5.0e-6,
                 'Contact':'Flat',
                 'Shape':'Rectangular'}
RW80L40P5p0CS = {'Name':'RW80L40P5p0CS',
                 'Length':40e-6, 
                 'Width':80e-6,
                 'Area':3200e-12,
                 'Pass':5.0e-6,
                 'Contact':'Flat',
                 'Shape':'Rectangular'}
RW80L80P5p0CS = {'Name':'RW80L80P5p0CS',
                 'Length':80e-6, 
                 'Width':80e-6,
                 'Area':6400e-12,
                 'Pass':5.0e-6,
                 'Contact':'Flat',
                 'Shape':'Rectangular'}
#circular/rectangulars S
CW120L5P5p0CS = {'Name':'CW120L5P5p0CS',
                 'Length':5e-6, 
                 'Width':120e-6,
                 'Area':1178e-12,
                 'Pass':5.0e-6,
                 'Contact':'Flat',
                 'Shape':'Circular'}
CW120L10P5p0CS = {'Name':'CW120L10P5p0CS',
                 'Length':10e-6, 
                 'Width':120e-6,
                 'Area':2199e-12,
                 'Pass':5.0e-6,
                 'Contact':'Flat',
                 'Shape':'Circular'}
CW120L18P5p0CS = {'Name':'CW120L18P5p0CS',
                 'Length':18e-6, 
                 'Width':120e-6,
                 'Area':3506e-12,
                 'Pass':5.0e-6,
                 'Contact':'Flat',
                 'Shape':'Circular'}
CW120L30P5p0CS = {'Name':'CW120L30P5p0CS',
                 'Length':30e-6, 
                 'Width':120e-6,
                 'Area':4712e-12,
                 'Pass':5.0e-6,
                 'Contact':'Flat',
                 'Shape':'Circular'}
CW120L40P5p0CS = {'Name':'CW120L40P5p0CS',
                 'Length':40e-6, 
                 'Width':120e-6,
                 'Area':5026e-12,
                 'Pass':5.0e-6,
                 'Contact':'Flat',
                 'Shape':'Circular'}
#circular/rectangulars E
RW120L5P5p0CS = {'Name':'RW120L5P5p0CS',
                 'Length':5e-6, 
                 'Width':120e-6,
                 'Area':600e-12,
                 'Pass':5.0e-6,
                 'Contact':'Flat',
                 'Shape':'Rectangular'}
RW120L10P5p0CS = {'Name':'RW120L10P5p0CS',
                 'Length':10e-6, 
                 'Width':120e-6,
                 'Area':1200e-12,
                 'Pass':5.0e-6,
                 'Contact':'Flat',
                 'Shape':'Rectangular'}
RW120L20P5p0CS = {'Name':'RW120L20P5p0CS',
                 'Length':20e-6, 
                 'Width':120e-6,
                 'Area':2400e-12,
                 'Pass':5.0e-6,
                 'Contact':'Flat',
                 'Shape':'Rectangular'}
RW120L60P5p0CS = {'Name':'RW120L60P5p0CS',
                 'Length':60e-6, 
                 'Width':120e-6,
                 'Area':7200e-12,
                 'Pass':5.0e-6,
                 'Contact':'Flat',
                 'Shape':'Rectangular'}
RW120L120P5p0CS = {'Name':'RW120L120P5p0CS',
                 'Length':120e-6, 
                 'Width':120e-6,
                 'Area':14400e-12,
                 'Pass':5.0e-6,
                 'Contact':'Flat',
                 'Shape':'Rectangular'}

#circular/rectangulars O
CW80L5P5p0CS = {'Name':'CW80L5P5p0CS',
                 'Length':5e-6, 
                 'Width':80e-6,
                 'Area':1178e-12,
                 'Pass':5.0e-6,
                 'Contact':'Flat',
                 'Shape':'Circular'}
CW80L8P5p0CS = {'Name':'CW80L8P5p0CS',
                 'Length':8e-6, 
                 'Width':80e-6,
                 'Area':1810e-12,
                 'Pass':5.0e-6,
                 'Contact':'Flat',
                 'Shape':'Circular'}
CW80L12P5p0CS = {'Name':'CW80L12P5p0CS',
                 'Length':12e-6, 
                 'Width':80e-6,
                 'Area':2564e-12,
                 'Pass':5.0e-6,
                 'Contact':'Flat',
                 'Shape':'Circular'}
CW80L16P5p0CS = {'Name':'CW80L16P5p0CS',
                 'Length':16e-6, 
                 'Width':80e-6,
                 'Area':3217e-12,
                 'Pass':5.0e-6,
                 'Contact':'Flat',
                 'Shape':'Circular'}
CW80L20P5p0CS = {'Name':'CW80L20P5p0CS',
                 'Length':20e-6, 
                 'Width':80e-6,
                 'Area':3770e-12,
                 'Pass':5.0e-6,
                 'Contact':'Flat',
                 'Shape':'Circular'}
#transparent
RW50L30P3p0CS = {'Name':'RW50L30P3p0CS',
                 'Length':50e-6, 
                 'Width':30e-6,
                 'Area':1500e-12,
                 'Pass':3e-6,
                 'Contact':'Flat',
                 'Shape':'Rectangular'}

RW50L30P3p0C1 = {'Name':'RW50L30P3p0C1',
                 'Length':50e-6, 
                 'Width':30e-6,
                 'Area':1500e-12,
                 'Pass':3e-6,
                 'Contact':'Flat Short-Common',
                 'Shape':'Rectangular'}

RW50L30P3p0C2 = {'Name':'RW50L30P3p0C2',
                 'Length':50e-6, 
                 'Width':30e-6,
                 'Area':1500e-12,
                 'Pass':3e-6,
                 'Contact':'Flat Long-Common',
                 'Shape':'Rectangular'}

RW50L30P3p0C3 = {'Name':'RW50L30P3p0C3',
                 'Length':50e-6, 
                 'Width':30e-6,
                 'Area':1500e-12,
                 'Pass':3e-6,
                 'Contact':'Flat Single',
                 'Shape':'Rectangular'}

RW50L30P3p0C4 = {'Name':'RW50L30P3p0C4',
                 'Length':50e-6, 
                 'Width':30e-6,
                 'Area':1500e-12,
                 'Pass':3e-6,
                 'Contact':'Flat Double',
                 'Shape':'Rectangular'}
#EEG
RW6000L6000P40CS = {'Name':'RW6000L6000P40CS',
                 'Length':6000e-6, 
                 'Width':6000e-6,
                 'Area':36000000e-12,
                 'Pass':40e-6,
                 'Contact':'Flat',
                 'Shape':'Rectangular'}
RW24000L1000P40CS = {'Name':'RW24000L1000P40CS',
                 'Length':1000e-6, 
                 'Width':24000e-6,
                 'Area':36000000e-12,
                 'Pass':40e-6,
                 'Contact':'Flat',
                 'Shape':'Rectangular'}
RW6000L1000P40CS = {'Name':'RW6000L1000P40CS',
                 'Length':1000e-6, 
                 'Width':6000e-6,
                 'Area':6000000e-12,
                 'Pass':40e-6,
                 'Contact':'Flat',
                 'Shape':'Rectangular'}
RW4000L6000PWCS = {'Name':'RW4000L6000PWCS',
                 'Length':6000e-6, 
                 'Width':4000e-6,
                 'Area':24000000e-12,
                 'Pass':1000,
                 'Contact':'Flat',
                 'Shape':'Rectangular'}


#==============================================================================
# Flat contact different length new
#==============================================================================

### Flat contact new N
RW40L40P1p5CS2 = {'Name':'RW40L40P1p5CS2',
                 'Length':40e-6, 
                 'Width':40e-6,
                 'Area':1600e-12,
                 'Pass':1.5e-6,
                 'Contact':'Flat2um',
                 'Shape':'Rectangular'}                 

RW40L20P1p5CS2 = {'Name':'RW40L20P1p5CS2',
                 'Length':20e-6, 
                 'Width':40e-6,
                 'Area':800e-12,
                 'Pass':1.5e-6,
                 'Contact':'Flat2um',
                 'Shape':'Rectangular'}                 
                 
RW40L10P1p5CS2 = {'Name':'RW40L10P1p5CS2',
                 'Length':10e-6, 
                 'Width':40e-6,
                 'Area':400e-12,
                 'Pass':1.5e-6,
                 'Contact':'Flat2um',
                 'Shape':'Rectangular'}                 

RW40L5P1p5CS2 = {'Name':'RW40L5P1p5CS2',
                 'Length':5e-6, 
                 'Width':40e-6,
                 'Area':200e-12,
                 'Pass':1.5e-6,
                 'Contact':'Flat2um',
                 'Shape':'Rectangular'}                 

RW40L2p5P1p5CS2 = {'Name':'RW40L2p5P1p5CS2',
                 'Length':2.5e-6, 
                 'Width':40e-6,
                 'Area':100e-12,
                 'Pass':1.5e-6,
                 'Contact':'Flat2um',
                 'Shape':'Rectangular'}   

### Flat contact new N no SU8
RW40L43P0CS2 = {'Name':'RW40L43P0CS2',
                 'Length':43e-6, 
                 'Width':40e-6,
                 'Area':1600e-12,
                 'Pass':0,
                 'Contact':'Flat2um',
                 'Shape':'Rectangular'}                 

RW40L23P0CS2 = {'Name':'RW40L23P0CS2',
                 'Length':23e-6, 
                 'Width':40e-6,
                 'Area':800e-12,
                 'Pass':0,
                 'Contact':'Flat2um',
                 'Shape':'Rectangular'}                 
                 
RW40L13P0CS2 = {'Name':'RW40L13P0CS2',
                 'Length':13e-6, 
                 'Width':40e-6,
                 'Area':400e-12,
                 'Pass':0,
                 'Contact':'Flat2um',
                 'Shape':'Rectangular'}                 

RW40L8P0CS2 = {'Name':'RW40L8P0CS2',
                 'Length':8e-6, 
                 'Width':40e-6,
                 'Area':200e-12,
                 'Pass':0,
                 'Contact':'Flat2um',
                 'Shape':'Rectangular'}                 

RW40L5p5P0CS2 = {'Name':'RW40L5p5P0CS2',
                 'Length':5.5e-6, 
                 'Width':40e-6,
                 'Area':100e-12,
                 'Pass':0,
                 'Contact':'Flat2um',
                 'Shape':'Rectangular'}    

### Flat 4um contacts E

RW40L40P1p5CS4 = {'Name':'RW40L40P1p5CS4',
                 'Length':40e-6, 
                 'Width':40e-6,
                 'Area':1600e-12,
                 'Pass':1.5e-6,
                 'Contact':'Flat4um',
                 'Shape':'Rectangular'}                 

RW40L20P1p5CS4 = {'Name':'RW40L20P1p5CS4',
                 'Length':20e-6, 
                 'Width':40e-6,
                 'Area':800e-12,
                 'Pass':1.5e-6,
                 'Contact':'Flat4um',
                 'Shape':'Rectangular'}                 

RW40L10P1p5CS4 = {'Name':'RW40L10P1p5CS4',
                 'Length':10e-6, 
                 'Width':40e-6,
                 'Area':400e-12,
                 'Pass':1.5e-6,
                 'Contact':'Flat4um',
                 'Shape':'Rectangular'}                 

RW40L5P1p5CS4 = {'Name':'RW40L5P1p5CS4',
                 'Length':5e-6, 
                 'Width':40e-6,
                 'Area':200e-12,
                 'Pass':1.5e-6,
                 'Contact':'Flat4um',
                 'Shape':'Rectangular'}                 

RW40L2p5P1p5CS4 = {'Name':'RW40L2p5P1p5CS4',
                 'Length':2.5e-6, 
                 'Width':40e-6,
                 'Area':100e-12,
                 'Pass':1.5e-6,
                 'Contact':'Flat4um',
                 'Shape':'Rectangular'} 

### Flat 4um contacts E No SU8

RW40L43P0CS4 = {'Name':'RW40L43P0CS4',
                 'Length':43e-6, 
                 'Width':40e-6,
                 'Area':1600e-12,
                 'Pass':0,
                 'Contact':'Flat4um',
                 'Shape':'Rectangular'}                 

RW40L23P0CS4 = {'Name':'RW40L23P0CS4',
                 'Length':23e-6, 
                 'Width':40e-6,
                 'Area':800e-12,
                 'Pass':0,
                 'Contact':'Flat4um',
                 'Shape':'Rectangular'}                 

RW40L13P0CS4 = {'Name':'RW40L13P0CS4',
                 'Length':13e-6, 
                 'Width':40e-6,
                 'Area':400e-12,
                 'Pass':0,
                 'Contact':'Flat4um',
                 'Shape':'Rectangular'}                 

RW40L8P0CS4 = {'Name':'RW40L8P0CS4',
                 'Length':8e-6, 
                 'Width':40e-6,
                 'Area':200e-12,
                 'Pass':0,
                 'Contact':'Flat4um',
                 'Shape':'Rectangular'}                 

RW40L5p5P0CS4 = {'Name':'RW40L5p5P0CS4',
                 'Length':5.5e-6, 
                 'Width':40e-6,
                 'Area':100e-12,
                 'Pass':0,
                 'Contact':'Flat4um',
                 'Shape':'Rectangular'} 

### Flat 8um contacts W

RW40L40P1p5CS8 = {'Name':'RW40L40P1p5CS8',
                 'Length':40e-6, 
                 'Width':40e-6,
                 'Area':1600e-12,
                 'Pass':1.5e-6,
                 'Contact':'Flat8um',
                 'Shape':'Rectangular'}                 

RW40L20P1p5CS8 = {'Name':'RW40L20P1p5CS8',
                 'Length':20e-6, 
                 'Width':40e-6,
                 'Area':800e-12,
                 'Pass':1.5e-6,
                 'Contact':'Flat8um',
                 'Shape':'Rectangular'}                 

RW40L10P1p5CS8 = {'Name':'RW40L10P1p5CS8',
                 'Length':10e-6, 
                 'Width':40e-6,
                 'Area':400e-12,
                 'Pass':1.5e-6,
                 'Contact':'Flat8um',
                 'Shape':'Rectangular'}                 

RW40L5P1p5CS8 = {'Name':'RW40L5P1p5CS8',
                 'Length':5e-6, 
                 'Width':40e-6,
                 'Area':200e-12,
                 'Pass':1.5e-6,
                 'Contact':'Flat8um',
                 'Shape':'Rectangular'}                 

RW40L2p5P1p5CS8 = {'Name':'RW40L2p5P1p5CS8',
                 'Length':2.5e-6, 
                 'Width':40e-6,
                 'Area':100e-12,
                 'Pass':1.5e-6,
                 'Contact':'Flat8um',
                 'Shape':'Rectangular'} 

### Flat 8um contacts W No SU8

RW40L43P0CS8 = {'Name':'RW40L43P0CS8',
                 'Length':43e-6, 
                 'Width':40e-6,
                 'Area':1600e-12,
                 'Pass':0,
                 'Contact':'Flat8um',
                 'Shape':'Rectangular'}                 

RW40L23P0CS8 = {'Name':'RW40L23P0CS8',
                 'Length':23e-6, 
                 'Width':40e-6,
                 'Area':800e-12,
                 'Pass':0,
                 'Contact':'Flat8um',
                 'Shape':'Rectangular'}                 

RW40L13P0CS8 = {'Name':'RW40L13P0CS8',
                 'Length':13e-6, 
                 'Width':40e-6,
                 'Area':400e-12,
                 'Pass':0,
                 'Contact':'Flat8um',
                 'Shape':'Rectangular'}                 

RW40L8P0CS8 = {'Name':'RW40L8P0CS8',
                 'Length':8e-6, 
                 'Width':40e-6,
                 'Area':200e-12,
                 'Pass':0,
                 'Contact':'Flat8um',
                 'Shape':'Rectangular'}                 

RW40L5p5P0CS8 = {'Name':'RW40L5p5P0CS8',
                 'Length':5.5e-6, 
                 'Width':40e-6,
                 'Area':100e-12,
                 'Pass':0,
                 'Contact':'Flat8um',
                 'Shape':'Rectangular'} 
      
### Flat 15um contacts S

RW40L40P1p5CS15 = {'Name':'RW40L40P1p5CS15',
                 'Length':40e-6, 
                 'Width':40e-6,
                 'Area':1600e-12,
                 'Pass':1.5e-6,
                 'Contact':'Flat15um',
                 'Shape':'Rectangular'}                 

RW40L20P1p5CS15 = {'Name':'RW40L20P1p5CS15',
                 'Length':20e-6, 
                 'Width':40e-6,
                 'Area':800e-12,
                 'Pass':1.5e-6,
                 'Contact':'Flat15um',
                 'Shape':'Rectangular'}                 

RW40L10P1p5CS15 = {'Name':'RW40L10P1p5CS15',
                 'Length':10e-6, 
                 'Width':40e-6,
                 'Area':400e-12,
                 'Pass':1.5e-6,
                 'Contact':'Flat15um',
                 'Shape':'Rectangular'}                 

RW40L5P1p5CS15 = {'Name':'RW40L5P1p5CS15',
                 'Length':5e-6, 
                 'Width':40e-6,
                 'Area':200e-12,
                 'Pass':1.5e-6,
                 'Contact':'Flat15um',
                 'Shape':'Rectangular'}                 

RW40L2p5P1p5CS15 = {'Name':'RW40L2p5P1p5CS15',
                 'Length':2.5e-6, 
                 'Width':40e-6,
                 'Area':100e-12,
                 'Pass':1.5e-6,
                 'Contact':'Flat15um',
                 'Shape':'Rectangular'} 

### Flat 15um contacts S No SU8

RW40L43P0CS15 = {'Name':'RW40L43P0CS15',
                 'Length':43e-6, 
                 'Width':40e-6,
                 'Area':1600e-12,
                 'Pass':0,
                 'Contact':'Flat15um',
                 'Shape':'Rectangular'}                 

RW40L23P0CS15 = {'Name':'RW40L23P0CS15',
                 'Length':23e-6, 
                 'Width':40e-6,
                 'Area':800e-12,
                 'Pass':0,
                 'Contact':'Flat15um',
                 'Shape':'Rectangular'}                 

RW40L13P0CS15 = {'Name':'RW40L13P0CS15',
                 'Length':13e-6, 
                 'Width':40e-6,
                 'Area':400e-12,
                 'Pass':0,
                 'Contact':'Flat15um',
                 'Shape':'Rectangular'}                 

RW40L8P0CS15 = {'Name':'RW40L8P0CS15',
                 'Length':8e-6, 
                 'Width':40e-6,
                 'Area':200e-12,
                 'Pass':0,
                 'Contact':'Flat15um',
                 'Shape':'Rectangular'}                 

RW40L5p5P0CS15 = {'Name':'RW40L5p5P0CS15',
                 'Length':5.5e-6, 
                 'Width':40e-6,
                 'Area':100e-12,
                 'Pass':0,
                 'Contact':'Flat15um',
                 'Shape':'Rectangular'} 

#==============================================================================
# Width chanegs X2 new
#==============================================================================

### Flat contact new N W=5um
RW5L40P1p5CS5 = {'Name':'RW5L40P1p5CS5',
                 'Length':40e-6, 
                 'Width':5e-6,
                 'Area':1600e-12,
                 'Pass':1.5e-6,
                 'Contact':'Flat5um',
                 'Shape':'Rectangular'}                 

RW5L20P1p5CS5 = {'Name':'RW5L20P1p5CS5',
                 'Length':20e-6, 
                 'Width':5e-6,
                 'Area':800e-12,
                 'Pass':1.5e-6,
                 'Contact':'Flat5um',
                 'Shape':'Rectangular'}                 
                 
RW5L10P1p5CS5 = {'Name':'RW5L10P1p5CS5',
                 'Length':10e-6, 
                 'Width':5e-6,
                 'Area':400e-12,
                 'Pass':1.5e-6,
                 'Contact':'Flat5um',
                 'Shape':'Rectangular'}                 

RW5L5P1p5CS5 = {'Name':'RW5L5P1p5CS5',
                 'Length':5e-6, 
                 'Width':5e-6,
                 'Area':200e-12,
                 'Pass':1.5e-6,
                 'Contact':'Flat5um',
                 'Shape':'Rectangular'}                 

RW5L2p5P1p5CS5 = {'Name':'RW5L2p5P1p5CS5',
                 'Length':2.5e-6, 
                 'Width':5e-6,
                 'Area':100e-12,
                 'Pass':1.5e-6,
                 'Contact':'Flat5um',
                 'Shape':'Rectangular'}   

### Flat contact new N W=5um no SU8
RW5L43P0CS5 = {'Name':'RW5L43P0CS5',
                 'Length':43e-6, 
                 'Width':5e-6,
                 'Area':1600e-12,
                 'Pass':0,
                 'Contact':'Flat5um',
                 'Shape':'Rectangular'}                 

RW5L23P0CS5 = {'Name':'RW5L23P0CS5',
                 'Length':23e-6, 
                 'Width':5e-6,
                 'Area':800e-12,
                 'Pass':0,
                 'Contact':'Flat5um',
                 'Shape':'Rectangular'}                 
                 
RW5L13P0CS5 = {'Name':'RW5L13P0CS5',
                 'Length':13e-6, 
                 'Width':5e-6,
                 'Area':400e-12,
                 'Pass':0,
                 'Contact':'Flat5um',
                 'Shape':'Rectangular'}                 

RW5L8P0CS5 = {'Name':'RW5L8P0CS5',
                 'Length':8e-6, 
                 'Width':5e-6,
                 'Area':200e-12,
                 'Pass':0,
                 'Contact':'Flat5um',
                 'Shape':'Rectangular'}                 

RW5L5p5P0CS5 = {'Name':'RW5L5p5P0CS5',
                 'Length':5.5e-6, 
                 'Width':5e-6,
                 'Area':100e-12,
                 'Pass':0,
                 'Contact':'Flat5um',
                 'Shape':'Rectangular'}    

### Flat contact new E W=10um
RW10L40P1p5CS5 = {'Name':'RW10L40P1p5CS5',
                 'Length':40e-6, 
                 'Width':10e-6,
                 'Area':1600e-12,
                 'Pass':1.5e-6,
                 'Contact':'Flat5um',
                 'Shape':'Rectangular'}                 

RW10L20P1p5CS5 = {'Name':'RW10L20P1p5CS5',
                 'Length':20e-6, 
                 'Width':10e-6,
                 'Area':800e-12,
                 'Pass':1.5e-6,
                 'Contact':'Flat5um',
                 'Shape':'Rectangular'}                 
                 
RW10L10P1p5CS5 = {'Name':'RW10L10P1p5CS5',
                 'Length':10e-6, 
                 'Width':10e-6,
                 'Area':400e-12,
                 'Pass':1.5e-6,
                 'Contact':'Flat5um',
                 'Shape':'Rectangular'}                 

RW10L5P1p5CS5 = {'Name':'RW10L5P1p5CS5',
                 'Length':5e-6, 
                 'Width':10e-6,
                 'Area':200e-12,
                 'Pass':1.5e-6,
                 'Contact':'Flat5um',
                 'Shape':'Rectangular'}                 

RW10L2p5P1p5CS5 = {'Name':'RW10L2p5P1p5CS5',
                 'Length':2.5e-6, 
                 'Width':10e-6,
                 'Area':100e-12,
                 'Pass':1.5e-6,
                 'Contact':'Flat5um',
                 'Shape':'Rectangular'}   

### Flat contact new E W=10um no SU8
RW10L43P0CS5 = {'Name':'RW10L43P0CS5',
                 'Length':43e-6, 
                 'Width':10e-6,
                 'Area':1600e-12,
                 'Pass':0,
                 'Contact':'Flat5um',
                 'Shape':'Rectangular'}                 

RW10L23P0CS5 = {'Name':'RW10L23P0CS5',
                 'Length':23e-6, 
                 'Width':10e-6,
                 'Area':800e-12,
                 'Pass':0,
                 'Contact':'Flat5um',
                 'Shape':'Rectangular'}                 
                 
RW10L13P0CS5 = {'Name':'RW10L13P0CS5',
                 'Length':13e-6, 
                 'Width':10e-6,
                 'Area':400e-12,
                 'Pass':0,
                 'Contact':'Flat5um',
                 'Shape':'Rectangular'}                 

RW10L8P0CS5 = {'Name':'RW10L8P0CS5',
                 'Length':8e-6, 
                 'Width':10e-6,
                 'Area':200e-12,
                 'Pass':0,
                 'Contact':'Flat5um',
                 'Shape':'Rectangular'}                 

RW10L5p5P0CS5 = {'Name':'RW10L5p5P0CS5',
                 'Length':5.5e-6, 
                 'Width':10e-6,
                 'Area':100e-12,
                 'Pass':0,
                 'Contact':'Flat5um',
                 'Shape':'Rectangular'}     

### Flat contact new W W=20um
RW20L40P1p5CS5 = {'Name':'RW20L40P1p5CS5',
                 'Length':40e-6, 
                 'Width':20e-6,
                 'Area':1600e-12,
                 'Pass':1.5e-6,
                 'Contact':'Flat5um',
                 'Shape':'Rectangular'}                 

RW20L20P1p5CS5 = {'Name':'RW20L20P1p5CS5',
                 'Length':20e-6, 
                 'Width':20e-6,
                 'Area':800e-12,
                 'Pass':1.5e-6,
                 'Contact':'Flat5um',
                 'Shape':'Rectangular'}                 
                 
RW20L10P1p5CS5 = {'Name':'RW20L10P1p5CS5',
                 'Length':10e-6, 
                 'Width':20e-6,
                 'Area':400e-12,
                 'Pass':1.5e-6,
                 'Contact':'Flat5um',
                 'Shape':'Rectangular'}                 

RW20L5P1p5CS5 = {'Name':'RW20L5P1p5CS5',
                 'Length':5e-6, 
                 'Width':20e-6,
                 'Area':200e-12,
                 'Pass':1.5e-6,
                 'Contact':'Flat5um',
                 'Shape':'Rectangular'}                 

RW20L2p5P1p5CS5 = {'Name':'RW20L2p5P1p5CS5',
                 'Length':2.5e-6, 
                 'Width':20e-6,
                 'Area':100e-12,
                 'Pass':1.5e-6,
                 'Contact':'Flat5um',
                 'Shape':'Rectangular'}   

### Flat contact new W W=20um no SU8
RW20L43P0CS5 = {'Name':'RW20L43P0CS5',
                 'Length':43e-6, 
                 'Width':20e-6,
                 'Area':1600e-12,
                 'Pass':0,
                 'Contact':'Flat5um',
                 'Shape':'Rectangular'}                 

RW20L23P0CS5 = {'Name':'RW20L23P0CS5',
                 'Length':23e-6, 
                 'Width':20e-6,
                 'Area':800e-12,
                 'Pass':0,
                 'Contact':'Flat5um',
                 'Shape':'Rectangular'}                 
                 
RW20L13P0CS5 = {'Name':'RW20L13P0CS5',
                 'Length':13e-6, 
                 'Width':20e-6,
                 'Area':400e-12,
                 'Pass':0,
                 'Contact':'Flat5um',
                 'Shape':'Rectangular'}                 

RW20L8P0CS5 = {'Name':'RW20L8P0CS5',
                 'Length':8e-6, 
                 'Width':20e-6,
                 'Area':200e-12,
                 'Pass':0,
                 'Contact':'Flat5um',
                 'Shape':'Rectangular'}                 

RW20L5p5P0CS5 = {'Name':'RW20L5p5P0CS5',
                 'Length':5.5e-6, 
                 'Width':20e-6,
                 'Area':100e-12,
                 'Pass':0,
                 'Contact':'Flat5um',
                 'Shape':'Rectangular'}  
      
### Flat contact new S W=40um
RW40L40P1p5CS5 = {'Name':'RW40L40P1p5CS5',
                 'Length':40e-6, 
                 'Width':40e-6,
                 'Area':1600e-12,
                 'Pass':1.5e-6,
                 'Contact':'Flat5um',
                 'Shape':'Rectangular'}                 

RW40L20P1p5CS5 = {'Name':'RW40L20P1p5CS5',
                 'Length':20e-6, 
                 'Width':40e-6,
                 'Area':800e-12,
                 'Pass':1.5e-6,
                 'Contact':'Flat5um',
                 'Shape':'Rectangular'}                 
                 
RW40L10P1p5CS5 = {'Name':'RW40L10P1p5CS5',
                 'Length':10e-6, 
                 'Width':40e-6,
                 'Area':400e-12,
                 'Pass':1.5e-6,
                 'Contact':'Flat5um',
                 'Shape':'Rectangular'}                 

RW40L5P1p5CS5 = {'Name':'RW40L5P1p5CS5',
                 'Length':5e-6, 
                 'Width':40e-6,
                 'Area':200e-12,
                 'Pass':1.5e-6,
                 'Contact':'Flat5um',
                 'Shape':'Rectangular'}                 

RW40L2p5P1p5CS5 = {'Name':'RW40L2p5P1p5CS5',
                 'Length':2.5e-6, 
                 'Width':40e-6,
                 'Area':100e-12,
                 'Pass':1.5e-6,
                 'Contact':'Flat5um',
                 'Shape':'Rectangular'}   

### Flat contact new S W=40um no SU8
RW40L43P0CS5 = {'Name':'RW40L43P0CS5',
                 'Length':43e-6, 
                 'Width':40e-6,
                 'Area':1600e-12,
                 'Pass':0,
                 'Contact':'Flat5um',
                 'Shape':'Rectangular'}                 

RW40L23P0CS5 = {'Name':'RW40L23P0CS5',
                 'Length':23e-6, 
                 'Width':40e-6,
                 'Area':800e-12,
                 'Pass':0,
                 'Contact':'Flat5um',
                 'Shape':'Rectangular'}                 
                 
RW40L13P0CS5 = {'Name':'RW40L13P0CS5',
                 'Length':13e-6, 
                 'Width':40e-6,
                 'Area':400e-12,
                 'Pass':0,
                 'Contact':'Flat5um',
                 'Shape':'Rectangular'}                 

RW40L8P0CS5 = {'Name':'RW40L8P0CS5',
                 'Length':8e-6, 
                 'Width':40e-6,
                 'Area':200e-12,
                 'Pass':0,
                 'Contact':'Flat5um',
                 'Shape':'Rectangular'}                 

RW40L5p5P0CS5 = {'Name':'RW40L5p5P0CS5',
                 'Length':5.5e-6, 
                 'Width':40e-6,
                 'Area':100e-12,
                 'Pass':0,
                 'Contact':'Flat5um',
                 'Shape':'Rectangular'}  

#==============================================================================
# X1 new
#==============================================================================
# N
RW40L16P0CS5 = {'Name':'RW40L16P0CS5',
                 'Length':16e-6, 
                 'Width':40e-6,
                 'Area':100e-12,
                 'Pass':0,
                 'Contact':'Flat5um',
                 'Shape':'Rectangular'} 
# E
RW40L7P8CS5 = {'Name':'RW40L7P8CS5',
                 'Length':7e-6, 
                 'Width':40e-6,
                 'Area':100e-12,
                 'Pass':8e-6,
                 'Contact':'Flat5um',
                 'Shape':'Rectangular'}
# W 
RW40L15P2CS5 = {'Name':'RW40L15P2CS5',
                 'Length':15e-6, 
                 'Width':40e-6,
                 'Area':100e-12,
                 'Pass':2e-6,
                 'Contact':'Flat5um',
                 'Shape':'Rectangular'} 
# S
RW40L5P7CS5 = {'Name':'RW40L5P7CS5',
                 'Length':5e-6, 
                 'Width':40e-6,
                 'Area':100e-12,
                 'Pass':7e-6,
                 'Contact':'Flat5um',
                 'Shape':'Rectangular'} 


EEGType={'NT1':None,
              'NT4':RW24000L1000P40CS,
              ## N standard
              'NT2':None,
              'NT3':None,
              'NT5':None,
              'NT6':None,                  
              ## N sTandard
              'NT7':None,
              'NT8':None,
              'NT9':None,
              'NT10':None,
              ## N standard
              'NT11':None,
              'NT12':None,
              'NT13':None,
              'NT14':None,
              ## N standard
              'NT15':None,
              'NT16':None,
              'NT17':None,
              'NT18':None,
              ## N standard
              'NT19':None,
              'NT20':None,
              'NT21':None,
              'NT22':None,               
              }

TransType={'ST1':RW50L30P3p0C1,
              'ST2':RW50L30P3p0C2,
              'ST3':RW50L30P3p0C3,
              'ST4':RW50L30P3p0C4,
              'ST5':RW50L30P3p0C2,
              'ST6':RW50L30P3p0C1,                  
              'ST7':RW50L30P3p0C4,
              'ST8':RW50L30P3p0C3,
              ## S None
              'ST9':None,
              'ST10':None,
              'ST11':None,
              'ST12':None,
              'ST13':None,
              'ST14':None,
              'ST15':None,
              'ST16':None,
              'ST17':None,
              'ST18':None,
              'ST19':None,
              'ST20':None,
              'ST21':None,
              'ST22':None,
              ## N standard
              'NT1':RW50L30P3p0C1,
              'NT2':RW50L30P3p0C2,
              'NT3':RW50L30P3p0C3,
              'NT4':RW50L30P3p0C4,
              'NT5':RW50L30P3p0C2,
              'NT6':RW50L30P3p0C1,                  
              'NT7':RW50L30P3p0C4,
              'NT8':RW50L30P3p0C3,
              ## N None
              'NT9':None,
              'NT10':None,
              'NT11':None,
              'NT12':None,
              'NT13':None,
              'NT14':None,
              'NT15':None,
              'NT16':None,
              'NT17':None,
              'NT18':None,
              'NT19':None,
              'NT20':None,
              'NT21':None,
              'NT22':None}

X2TrtTypes = {'ST1':None,
              'ST4':None,
              'ST19':RW40L2p5P1p5CC5,
              'ST20':RW40L2p5P1p5CC5,
              'ST21':RW40L2p5P1p5CC5,
              'ST22':RW40L2p5P1p5CC5,                  
              ## S circles    
              'ST15':RW40L5P1p5CC5,
              'ST16':RW40L5P1p5CC5,
              'ST17':RW40L5P1p5CC5,
              'ST18':RW40L5P1p5CC5,
              ## S circles    
              'ST11':RW40L10P1p5CC5,
              'ST12':RW40L10P1p5CC5,
              'ST13':RW40L10P1p5CC5,
              'ST14':RW40L10P1p5CC5,
              ## S circles                        
              'ST7':RW40L20P1p5CC5,
              'ST8':RW40L20P1p5CC5,
              'ST9':RW40L20P1p5CC5,
              'ST10':RW40L20P1p5CC5,
              ## S circles    
              'ST2':RW40L40P1p5CC5,
              'ST3':RW40L40P1p5CC5,
              'ST5':RW40L40P1p5CC5,
              'ST6':RW40L40P1p5CC5,
              ## N none
              'NT1':None,
              'NT4':None,
              ## N standard
              'NT2':RW40L2p5P1p5CS,
              'NT3':RW40L2p5P1p5CS,
              'NT5':RW40L2p5P1p5CS,
              'NT6':RW40L2p5P1p5CS,                  
              ## N sTandard
              'NT7':RW40L5P1p5CS,
              'NT8':RW40L5P1p5CS,
              'NT9':RW40L5P1p5CS,
              'NT10':RW40L5P1p5CS,
              ## N standard
              'NT11':RW40L10P1p5CS,
              'NT12':RW40L10P1p5CS,
              'NT13':RW40L10P1p5CS,
              'NT14':RW40L10P1p5CS,
              ## N standard
              'NT15':RW40L20P1p5CS,
              'NT16':RW40L20P1p5CS,
              'NT17':RW40L20P1p5CS,
              'NT18':RW40L20P1p5CS,
              ## N standard
              'NT19':RW40L40P1p5CS,
              'NT20':RW40L40P1p5CS,
              'NT21':RW40L40P1p5CS,
              'NT22':RW40L40P1p5CS,
              ## E None
              'ET11':None,
              'ET12':None,
              ## E Finger 2
              'ET1':RW40L2p5P1p5CF2,
              'ET6':RW40L2p5P1p5CF2,
              'ET17':RW40L2p5P1p5CF2,
              'ET22':RW40L2p5P1p5CF2,                
              ## E Finger 2
              'ET2':RW40L5P1p5CF2,
              'ET7':RW40L5P1p5CF2,
              'ET16':RW40L5P1p5CF2,
              'ET21':RW40L5P1p5CF2,
              ## E Finger 2
              'ET3':RW40L10P1p5CF2,
              'ET8':RW40L10P1p5CF2,
              'ET15':RW40L10P1p5CF2,
              'ET20':RW40L10P1p5CF2,
              ## E Finger 2
              'ET4':RW40L20P1p5CF2,
              'ET9':RW40L20P1p5CF2,
              'ET14':RW40L20P1p5CF2,
              'ET19':RW40L20P1p5CF2,
              ## E Finger 2
              'ET5':RW40L40P1p5CF2,
              'ET10':RW40L40P1p5CF2,
              'ET13':RW40L40P1p5CF2,
              'ET18':RW40L40P1p5CF2,
              ## O None
              'OT11':None,   
              'OT12':None,                      
              ## O Finger 3.5
              'OT5':RW40L2p5P1p5CF3p5,
              'OT10':RW40L2p5P1p5CF3p5,
              'OT13':RW40L2p5P1p5CF3p5,
              'OT18':RW40L2p5P1p5CF3p5,                
              ## O Finger 3.5
              'OT4':RW40L5P1p5CF3p5,
              'OT9':RW40L5P1p5CF3p5,
              'OT14':RW40L5P1p5CF3p5,
              'OT19':RW40L5P1p5CF3p5,
              ## O Finger 3.5
              'OT3':RW40L10P1p5CF3p5,
              'OT8':RW40L10P1p5CF3p5,
              'OT15':RW40L10P1p5CF3p5,
              'OT20':RW40L10P1p5CF3p5,
              ## O Finger 3.5
              'OT2':RW40L20P1p5CF3p5,
              'OT7':RW40L20P1p5CF3p5,
              'OT16':RW40L20P1p5CF3p5,
              'OT21':RW40L20P1p5CF3p5,
              ## O Finger 3.5
              'OT22':RW40L40P1p5CF3p5,
              'OT17':RW40L40P1p5CF3p5,
              'OT6':RW40L40P1p5CF3p5,
              'OT1':RW40L40P1p5CF3p5,                
              }
                     
X1TrtTypes = {'ST1':None,
              'ST4':None,
              'ST19':RW40L15P2p0CSOv,
              'ST20':RW40L15P2p0CSOv,
              'ST21':RW40L15P2p0CSOv,
              'ST22':RW40L15P2p0CSOv,                  
              ## S circles    
              'ST15':RW40L15P2p0CSOv,
              'ST16':RW40L15P2p0CSOv,
              'ST17':RW40L15P2p0CSOv,
              'ST18':RW40L15P2p0CSOv,
              ## S circles    
              'ST11':RW40L15P2p0CSOv,
              'ST12':RW40L15P2p0CSOv,
              'ST13':RW40L15P2p0CSOv,
              'ST14':RW40L15P2p0CSOv,
              ## S circles                        
              'ST7':RW40L15P2p0CSOv,
              'ST8':RW40L15P2p0CSOv,
              'ST9':RW40L15P2p0CSOv,
              'ST10':RW40L15P2p0CSOv,
              ## S circles    
              'ST2':RW40L15P2p0CSOv,
              'ST3':RW40L15P2p0CSOv,
              'ST5':RW40L15P2p0CSOv,
              'ST6':RW40L15P2p0CSOv,
              ## N none
              'NT1':None,
              'NT4':None,
              ## N standard
              'NT2':RW40L15P0p5CS,
              'NT3':RW40L15P0p5CS,
              'NT5':RW40L15P0p5CS,
              'NT6':RW40L15P0p5CS,                  
              ## N sTandard
              'NT7':RW40L15P0p5CS,
              'NT8':RW40L15P0p5CS,
              'NT9':RW40L15P0p5CS,
              'NT10':RW40L15P0p5CS,
              ## N standard
              'NT11':RW40L15P0p5CS,
              'NT12':RW40L15P0p5CS,
              'NT13':RW40L15P0p5CS,
              'NT14':RW40L15P0p5CS,
              ## N standard
              'NT15':RW40L15P0p5CS,
              'NT16':RW40L15P0p5CS,
              'NT17':RW40L15P0p5CS,
              'NT18':RW40L15P0p5CS,
              ## N standard
              'NT19':RW40L15P0p5CS,
              'NT20':RW40L15P0p5CS,
              'NT21':RW40L15P0p5CS,
              'NT22':RW40L15P0p5CS,
              ## E None
              'ET11':None,
              'ET12':None,
              ## E Finger 2
              'ET1':RW40L15P1p0CS,
              'ET6':RW40L15P1p0CS,
              'ET17':RW40L15P1p0CS,
              'ET22':RW40L15P1p0CS,                
              ## E Finger 2
              'ET2':RW40L15P1p0CS,
              'ET7':RW40L15P1p0CS,
              'ET16':RW40L15P1p0CS,
              'ET21':RW40L15P1p0CS,
              ## E Finger 2
              'ET3':RW40L15P1p0CS,
              'ET8':RW40L15P1p0CS,
              'ET15':RW40L15P1p0CS,
              'ET20':RW40L15P1p0CS,
              ## E Finger 2
              'ET4':RW40L15P1p0CS,
              'ET9':RW40L15P1p0CS,
              'ET14':RW40L15P1p0CS,
              'ET19':RW40L15P1p0CS,
              ## E Finger 2
              'ET5':RW40L15P1p0CS,
              'ET10':RW40L15P1p0CS,
              'ET13':RW40L15P1p0CS,
              'ET18':RW40L15P1p0CS,
              ## O None
              'OT11':None,   
              'OT12':None,                      
              ## O Finger 3.5
              'OT5':RW40L15P2p0CS,
              'OT10':RW40L15P2p0CS,
              'OT13':RW40L15P2p0CS,
              'OT18':RW40L15P2p0CS,                
              ## O Finger 3.5
              'OT4':RW40L15P2p0CS,
              'OT9':RW40L15P2p0CS,
              'OT14':RW40L15P2p0CS,
              'OT19':RW40L15P2p0CS,
              ## O Finger 3.5
              'OT3':RW40L15P2p0CS,
              'OT8':RW40L15P2p0CS,
              'OT15':RW40L15P2p0CS,
              'OT20':RW40L15P2p0CS,
              ## O Finger 3.5
              'OT2':RW40L15P2p0CS,
              'OT7':RW40L15P2p0CS,
              'OT16':RW40L15P2p0CS,
              'OT21':RW40L15P2p0CS,
              ## O Finger 3.5
              'OT22':RW40L15P2p0CS,
              'OT17':RW40L15P2p0CS,
              'OT6':RW40L15P2p0CS,
              'OT1':RW40L15P2p0CS,                
              }

X1TrtTypesNew = {'ST1':None,
              'ST4':None,
              'ST19':RW40L5P7CS5,
              'ST20':RW40L5P7CS5,
              'ST21':RW40L5P7CS5,
              'ST22':RW40L5P7CS5,                  
              ## S circles    
              'ST15':RW40L5P7CS5,
              'ST16':RW40L5P7CS5,
              'ST17':RW40L5P7CS5,
              'ST18':RW40L5P7CS5,
              ## S circles    
              'ST11':RW40L5P7CS5,
              'ST12':RW40L5P7CS5,
              'ST13':RW40L5P7CS5,
              'ST14':RW40L5P7CS5,
              ## S circles                        
              'ST7':RW40L5P7CS5,
              'ST8':RW40L5P7CS5,
              'ST9':RW40L5P7CS5,
              'ST10':RW40L5P7CS5,
              ## S circles    
              'ST2':RW40L5P7CS5,
              'ST3':RW40L5P7CS5,
              'ST5':RW40L5P7CS5,
              'ST6':RW40L5P7CS5,
              ## N none
              'NT1':None,
              'NT4':None,
              ## N standard
              'NT2':RW40L16P0CS5,
              'NT3':RW40L16P0CS5,
              'NT5':RW40L16P0CS5,
              'NT6':RW40L16P0CS5,                  
              ## N sTandard
              'NT7':RW40L16P0CS5,
              'NT8':RW40L16P0CS5,
              'NT9':RW40L16P0CS5,
              'NT10':RW40L16P0CS5,
              ## N standard
              'NT11':RW40L16P0CS5,
              'NT12':RW40L16P0CS5,
              'NT13':RW40L16P0CS5,
              'NT14':RW40L16P0CS5,
              ## N standard
              'NT15':RW40L16P0CS5,
              'NT16':RW40L16P0CS5,
              'NT17':RW40L16P0CS5,
              'NT18':RW40L16P0CS5,
              ## N standard
              'NT19':RW40L16P0CS5,
              'NT20':RW40L16P0CS5,
              'NT21':RW40L16P0CS5,
              'NT22':RW40L16P0CS5,
              ## E None
              'ET11':None,
              'ET12':None,
              ## E Finger 2
              'ET1':RW40L7P8CS5,
              'ET6':RW40L7P8CS5,
              'ET17':RW40L7P8CS5,
              'ET22':RW40L7P8CS5,                
              ## E Finger 2
              'ET2':RW40L7P8CS5,
              'ET7':RW40L7P8CS5,
              'ET16':RW40L7P8CS5,
              'ET21':RW40L7P8CS5,
              ## E Finger 2
              'ET3':RW40L7P8CS5,
              'ET8':RW40L7P8CS5,
              'ET15':RW40L7P8CS5,
              'ET20':RW40L7P8CS5,
              ## E Finger 2
              'ET4':RW40L7P8CS5,
              'ET9':RW40L7P8CS5,
              'ET14':RW40L7P8CS5,
              'ET19':RW40L7P8CS5,
              ## E Finger 2
              'ET5':RW40L7P8CS5,
              'ET10':RW40L7P8CS5,
              'ET13':RW40L7P8CS5,
              'ET18':RW40L7P8CS5,
              ## O None
              'OT11':None,   
              'OT12':None,                      
              ## O Finger 3.5
              'OT5':RW40L15P2CS5,
              'OT10':RW40L15P2CS5,
              'OT13':RW40L15P2CS5,
              'OT18':RW40L15P2CS5,                
              ## O Finger 3.5
              'OT4':RW40L15P2CS5,
              'OT9':RW40L15P2CS5,
              'OT14':RW40L15P2CS5,
              'OT19':RW40L15P2CS5,
              ## O Finger 3.5
              'OT3':RW40L15P2CS5,
              'OT8':RW40L15P2CS5,
              'OT15':RW40L15P2CS5,
              'OT20':RW40L15P2CS5,
              ## O Finger 3.5
              'OT2':RW40L15P2CS5,
              'OT7':RW40L15P2CS5,
              'OT16':RW40L15P2CS5,
              'OT21':RW40L15P2CS5,
              ## O Finger 3.5
              'OT22':RW40L15P2CS5,
              'OT17':RW40L15P2CS5,
              'OT6':RW40L15P2CS5,
              'OT1':RW40L15P2CS5,                
              }

X2TrtTypesContacts = {'ST1':None,
              'ST4':None,
              'ST19':RW40L2p5P1p5CS15,
              'ST20':RW40L2p5P1p5CS15,
              'ST21':RW40L2p5P1p5CS15,
              'ST22':RW40L2p5P1p5CS15,                  
              ## S   
              'ST15':RW40L5P1p5CS15,
              'ST16':RW40L5P1p5CS15,
              'ST17':RW40L5P1p5CS15,
              'ST18':RW40L5P1p5CS15,
              ## S   
              'ST11':RW40L10P1p5CS15,
              'ST12':RW40L10P1p5CS15,
              'ST13':RW40L10P1p5CS15,
              'ST14':RW40L10P1p5CS15,
              ## S                      
              'ST7':RW40L20P1p5CS15,
              'ST8':RW40L20P1p5CS15,
              'ST9':RW40L20P1p5CS15,
              'ST10':RW40L20P1p5CS15,
              ## S   
              'ST2':RW40L40P1p5CS15,
              'ST3':RW40L40P1p5CS15,
              'ST5':RW40L40P1p5CS15,
              'ST6':RW40L40P1p5CS15,
              ## N none
              'NT1':None,
              'NT4':None,
              ## N 
              'NT2':RW40L2p5P1p5CS2,
              'NT3':RW40L2p5P1p5CS2,
              'NT5':RW40L2p5P1p5CS2,
              'NT6':RW40L2p5P1p5CS2,                  
              ## N 
              'NT7':RW40L5P1p5CS2,
              'NT8':RW40L5P1p5CS2,
              'NT9':RW40L5P1p5CS2,
              'NT10':RW40L5P1p5CS2,
              ## N 
              'NT11':RW40L10P1p5CS2,
              'NT12':RW40L10P1p5CS2,
              'NT13':RW40L10P1p5CS2,
              'NT14':RW40L10P1p5CS2,
              ## N
              'NT15':RW40L20P1p5CS2,
              'NT16':RW40L20P1p5CS2,
              'NT17':RW40L20P1p5CS2,
              'NT18':RW40L20P1p5CS2,
              ## N 
              'NT19':RW40L40P1p5CS2,
              'NT20':RW40L40P1p5CS2,
              'NT21':RW40L40P1p5CS2,
              'NT22':RW40L40P1p5CS2,
              ## E None
              'ET11':None,
              'ET12':None,
              ## E 
              'ET1':RW40L2p5P1p5CS4,
              'ET6':RW40L2p5P1p5CS4,
              'ET17':RW40L2p5P1p5CS4,
              'ET22':RW40L2p5P1p5CS4,                
              ## E 
              'ET2':RW40L5P1p5CS4,
              'ET7':RW40L5P1p5CS4,
              'ET16':RW40L5P1p5CS4,
              'ET21':RW40L5P1p5CS4,
              ## E 
              'ET3':RW40L10P1p5CS4,
              'ET8':RW40L10P1p5CS4,
              'ET15':RW40L10P1p5CS4,
              'ET20':RW40L10P1p5CS4,
              ## E 
              'ET4':RW40L20P1p5CS4,
              'ET9':RW40L20P1p5CS4,
              'ET14':RW40L20P1p5CS4,
              'ET19':RW40L20P1p5CS4,
              ## E 
              'ET5':RW40L40P1p5CS4,
              'ET10':RW40L40P1p5CS4,
              'ET13':RW40L40P1p5CS4,
              'ET18':RW40L40P1p5CS4,
              ## O None
              'OT11':None,   
              'OT12':None,                      
              ## O 
              'OT5':RW40L2p5P1p5CS8,
              'OT10':RW40L2p5P1p5CS8,
              'OT13':RW40L2p5P1p5CS8,
              'OT18':RW40L2p5P1p5CS8,                
              ## O 
              'OT4':RW40L5P1p5CS8,
              'OT9':RW40L5P1p5CS8,
              'OT14':RW40L5P1p5CS8,
              'OT19':RW40L5P1p5CS8,
              ## O 
              'OT3':RW40L10P1p5CS8,
              'OT8':RW40L10P1p5CS8,
              'OT15':RW40L10P1p5CS8,
              'OT20':RW40L10P1p5CS8,
              ## O 
              'OT2':RW40L20P1p5CS8,
              'OT7':RW40L20P1p5CS8,
              'OT16':RW40L20P1p5CS8,
              'OT21':RW40L20P1p5CS8,
              ## O 
              'OT22':RW40L40P1p5CS8,
              'OT17':RW40L40P1p5CS8,
              'OT6':RW40L40P1p5CS8,
              'OT1':RW40L40P1p5CS8,                
              }

X2TrtTypesContactsNoSu8 = {'ST1':None,
              'ST4':None,
              'ST19':RW40L5p5P0CS15,
              'ST20':RW40L5p5P0CS15,
              'ST21':RW40L5p5P0CS15,
              'ST22':RW40L5p5P0CS15,                  
              ## S   
              'ST15':RW40L8P0CS15,
              'ST16':RW40L8P0CS15,
              'ST17':RW40L8P0CS15,
              'ST18':RW40L8P0CS15,
              ## S   
              'ST11':RW40L13P0CS15,
              'ST12':RW40L13P0CS15,
              'ST13':RW40L13P0CS15,
              'ST14':RW40L13P0CS15,
              ## S                      
              'ST7':RW40L23P0CS15,
              'ST8':RW40L23P0CS15,
              'ST9':RW40L23P0CS15,
              'ST10':RW40L23P0CS15,
              ## S   
              'ST2':RW40L43P0CS15,
              'ST3':RW40L43P0CS15,
              'ST5':RW40L43P0CS15,
              'ST6':RW40L43P0CS15,
              ## N none
              'NT1':None,
              'NT4':None,
              ## N 
              'NT2':RW40L5p5P0CS2,
              'NT3':RW40L5p5P0CS2,
              'NT5':RW40L5p5P0CS2,
              'NT6':RW40L5p5P0CS2,                  
              ## N 
              'NT7':RW40L8P0CS2,
              'NT8':RW40L8P0CS2,
              'NT9':RW40L8P0CS2,
              'NT10':RW40L8P0CS2,
              ## N 
              'NT11':RW40L13P0CS2,
              'NT12':RW40L13P0CS2,
              'NT13':RW40L13P0CS2,
              'NT14':RW40L13P0CS2,
              ## N
              'NT15':RW40L23P0CS2,
              'NT16':RW40L23P0CS2,
              'NT17':RW40L23P0CS2,
              'NT18':RW40L23P0CS2,
              ## N 
              'NT19':RW40L43P0CS2,
              'NT20':RW40L43P0CS2,
              'NT21':RW40L43P0CS2,
              'NT22':RW40L43P0CS2,
              ## E None
              'ET11':None,
              'ET12':None,
              ## E 
              'ET1':RW40L5p5P0CS4,
              'ET6':RW40L5p5P0CS4,
              'ET17':RW40L5p5P0CS4,
              'ET22':RW40L5p5P0CS4,                
              ## E 
              'ET2':RW40L8P0CS4,
              'ET7':RW40L8P0CS4,
              'ET16':RW40L8P0CS4,
              'ET21':RW40L8P0CS4,
              ## E 
              'ET3':RW40L13P0CS4,
              'ET8':RW40L13P0CS4,
              'ET15':RW40L13P0CS4,
              'ET20':RW40L13P0CS4,
              ## E 
              'ET4':RW40L23P0CS4,
              'ET9':RW40L23P0CS4,
              'ET14':RW40L23P0CS4,
              'ET19':RW40L23P0CS4,
              ## E 
              'ET5':RW40L43P0CS4,
              'ET10':RW40L43P0CS4,
              'ET13':RW40L43P0CS4,
              'ET18':RW40L43P0CS4,
              ## O None
              'OT11':None,   
              'OT12':None,                      
              ## O 
              'OT5':RW40L5p5P0CS8,
              'OT10':RW40L5p5P0CS8,
              'OT13':RW40L5p5P0CS8,
              'OT18':RW40L5p5P0CS8,                
              ## O 
              'OT4':RW40L8P0CS8,
              'OT9':RW40L8P0CS8,
              'OT14':RW40L8P0CS8,
              'OT19':RW40L8P0CS8,
              ## O 
              'OT3':RW40L13P0CS8,
              'OT8':RW40L13P0CS8,
              'OT15':RW40L13P0CS8,
              'OT20':RW40L13P0CS8,
              ## O 
              'OT2':RW40L23P0CS8,
              'OT7':RW40L23P0CS8,
              'OT16':RW40L23P0CS8,
              'OT21':RW40L23P0CS8,
              ## O 
              'OT22':RW40L43P0CS8,
              'OT17':RW40L43P0CS8,
              'OT6':RW40L43P0CS8,
              'OT1':RW40L43P0CS8,                
              }

X2TrtTypesWidth = {'ST1':None,
              'ST4':None,
              'ST19':RW40L2p5P1p5CS5,
              'ST20':RW40L2p5P1p5CS5,
              'ST21':RW40L2p5P1p5CS5,
              'ST22':RW40L2p5P1p5CS5,                  
              ## S   
              'ST15':RW40L5P1p5CS5,
              'ST16':RW40L5P1p5CS5,
              'ST17':RW40L5P1p5CS5,
              'ST18':RW40L5P1p5CS5,
              ## S   
              'ST11':RW40L10P1p5CS5,
              'ST12':RW40L10P1p5CS5,
              'ST13':RW40L10P1p5CS5,
              'ST14':RW40L10P1p5CS5,
              ## S                      
              'ST7':RW40L20P1p5CS5,
              'ST8':RW40L20P1p5CS5,
              'ST9':RW40L20P1p5CS5,
              'ST10':RW40L20P1p5CS5,
              ## S   
              'ST2':RW40L40P1p5CS5,
              'ST3':RW40L40P1p5CS5,
              'ST5':RW40L40P1p5CS5,
              'ST6':RW40L40P1p5CS5,
              ## N none
              'NT1':None,
              'NT4':None,
              ## N 
              'NT2':RW5L2p5P1p5CS5,
              'NT3':RW5L2p5P1p5CS5,
              'NT5':RW5L2p5P1p5CS5,
              'NT6':RW5L2p5P1p5CS5,                  
              ## N 
              'NT7':RW5L5P1p5CS5,
              'NT8':RW5L5P1p5CS5,
              'NT9':RW5L5P1p5CS5,
              'NT10':RW5L5P1p5CS5,
              ## N 
              'NT11':RW5L10P1p5CS5,
              'NT12':RW5L10P1p5CS5,
              'NT13':RW5L10P1p5CS5,
              'NT14':RW5L10P1p5CS5,
              ## N
              'NT15':RW5L20P1p5CS5,
              'NT16':RW5L20P1p5CS5,
              'NT17':RW5L20P1p5CS5,
              'NT18':RW5L20P1p5CS5,
              ## N 
              'NT19':RW5L40P1p5CS5,
              'NT20':RW5L40P1p5CS5,
              'NT21':RW5L40P1p5CS5,
              'NT22':RW5L40P1p5CS5,
              ## E None
              'ET11':None,
              'ET12':None,
              ## E 
              'ET1':RW10L2p5P1p5CS5,
              'ET6':RW10L2p5P1p5CS5,
              'ET17':RW10L2p5P1p5CS5,
              'ET22':RW10L2p5P1p5CS5,                
              ## E 
              'ET2':RW10L5P1p5CS5,
              'ET7':RW10L5P1p5CS5,
              'ET16':RW10L5P1p5CS5,
              'ET21':RW10L5P1p5CS5,
              ## E 
              'ET3':RW10L10P1p5CS5,
              'ET8':RW10L10P1p5CS5,
              'ET15':RW10L10P1p5CS5,
              'ET20':RW10L10P1p5CS5,
              ## E 
              'ET4':RW10L20P1p5CS5,
              'ET9':RW10L20P1p5CS5,
              'ET14':RW10L20P1p5CS5,
              'ET19':RW10L20P1p5CS5,
              ## E 
              'ET5':RW10L40P1p5CS5,
              'ET10':RW10L40P1p5CS5,
              'ET13':RW10L40P1p5CS5,
              'ET18':RW10L40P1p5CS5,
              ## O None
              'OT11':None,   
              'OT12':None,                      
              ## O 
              'OT5':RW20L2p5P1p5CS5,
              'OT10':RW20L2p5P1p5CS5,
              'OT13':RW20L2p5P1p5CS5,
              'OT18':RW20L2p5P1p5CS5,                
              ## O 
              'OT4':RW20L5P1p5CS5,
              'OT9':RW20L5P1p5CS5,
              'OT14':RW20L5P1p5CS5,
              'OT19':RW20L5P1p5CS5,
              ## O 
              'OT3':RW20L10P1p5CS5,
              'OT8':RW20L10P1p5CS5,
              'OT15':RW20L10P1p5CS5,
              'OT20':RW20L10P1p5CS5,
              ## O 
              'OT2':RW20L20P1p5CS5,
              'OT7':RW20L20P1p5CS5,
              'OT16':RW20L20P1p5CS5,
              'OT21':RW20L20P1p5CS5,
              ## O 
              'OT22':RW20L40P1p5CS5,
              'OT17':RW20L40P1p5CS5,
              'OT6':RW20L40P1p5CS5,
              'OT1':RW20L40P1p5CS5,                
              }

X2TrtTypesWidthNoSu8 = {'ST1':None,
              'ST4':None,
              'ST19':RW40L5p5P0CS5,
              'ST20':RW40L5p5P0CS5,
              'ST21':RW40L5p5P0CS5,
              'ST22':RW40L5p5P0CS5,                  
              ## S   
              'ST15':RW40L8P0CS5,
              'ST16':RW40L8P0CS5,
              'ST17':RW40L8P0CS5,
              'ST18':RW40L8P0CS5,
              ## S   
              'ST11':RW40L13P0CS5,
              'ST12':RW40L13P0CS5,
              'ST13':RW40L13P0CS5,
              'ST14':RW40L13P0CS5,
              ## S                      
              'ST7':RW40L23P0CS5,
              'ST8':RW40L23P0CS5,
              'ST9':RW40L23P0CS5,
              'ST10':RW40L23P0CS5,
              ## S   
              'ST2':RW40L43P0CS5,
              'ST3':RW40L43P0CS5,
              'ST5':RW40L43P0CS5,
              'ST6':RW40L43P0CS5,
              ## N none
              'NT1':None,
              'NT4':None,
              ## N 
              'NT2':RW5L5p5P0CS5,
              'NT3':RW5L5p5P0CS5,
              'NT5':RW5L5p5P0CS5,
              'NT6':RW5L5p5P0CS5,                  
              ## N 
              'NT7':RW5L8P0CS5,
              'NT8':RW5L8P0CS5,
              'NT9':RW5L8P0CS5,
              'NT10':RW5L8P0CS5,
              ## N 
              'NT11':RW5L13P0CS5,
              'NT12':RW5L13P0CS5,
              'NT13':RW5L13P0CS5,
              'NT14':RW5L13P0CS5,
              ## N
              'NT15':RW5L23P0CS5,
              'NT16':RW5L23P0CS5,
              'NT17':RW5L23P0CS5,
              'NT18':RW5L23P0CS5,
              ## N 
              'NT19':RW5L43P0CS5,
              'NT20':RW5L43P0CS5,
              'NT21':RW5L43P0CS5,
              'NT22':RW5L43P0CS5,
              ## E None
              'ET11':None,
              'ET12':None,
              ## E 
              'ET1':RW10L5p5P0CS5,
              'ET6':RW10L5p5P0CS5,
              'ET17':RW10L5p5P0CS5,
              'ET22':RW10L5p5P0CS5,                
              ## E 
              'ET2':RW10L8P0CS5,
              'ET7':RW10L8P0CS5,
              'ET16':RW10L8P0CS5,
              'ET21':RW10L8P0CS5,
              ## E 
              'ET3':RW10L13P0CS5,
              'ET8':RW10L13P0CS5,
              'ET15':RW10L13P0CS5,
              'ET20':RW10L13P0CS5,
              ## E 
              'ET4':RW10L23P0CS5,
              'ET9':RW10L23P0CS5,
              'ET14':RW10L23P0CS5,
              'ET19':RW10L23P0CS5,
              ## E 
              'ET5':RW10L43P0CS5,
              'ET10':RW10L43P0CS5,
              'ET13':RW10L43P0CS5,
              'ET18':RW10L43P0CS5,
              ## O None
              'OT11':None,   
              'OT12':None,                      
              ## O 
              'OT5':RW20L5p5P0CS5,
              'OT10':RW20L5p5P0CS5,
              'OT13':RW20L5p5P0CS5,
              'OT18':RW20L5p5P0CS5,                
              ## O 
              'OT4':RW20L8P0CS5,
              'OT9':RW20L8P0CS5,
              'OT14':RW20L8P0CS5,
              'OT19':RW20L8P0CS5,
              ## O 
              'OT3':RW20L13P0CS5,
              'OT8':RW20L13P0CS5,
              'OT15':RW20L13P0CS5,
              'OT20':RW20L13P0CS5,
              ## O 
              'OT2':RW20L23P0CS5,
              'OT7':RW20L23P0CS5,
              'OT16':RW20L23P0CS5,
              'OT21':RW20L23P0CS5,
              ## O 
              'OT22':RW20L43P0CS5,
              'OT17':RW20L43P0CS5,
              'OT6':RW20L43P0CS5,
              'OT1':RW20L43P0CS5,                
              }

CirTrtTypes = {'ST1':None,
              'ST18':CW120L40P5p0CS,
              'ST19':CW120L40P5p0CS,
              'ST20':CW120L40P5p0CS,
              'ST21':CW120L40P5p0CS,
              'ST22':None,                  
              ## S circles 
              'ST14':CW120L30P5p0CS,
              'ST15':CW120L30P5p0CS,
              'ST16':CW120L30P5p0CS,
              'ST17':CW120L30P5p0CS,
              ## S circles 
              'ST10':CW120L18P5p0CS,
              'ST11':CW120L18P5p0CS,
              'ST12':CW120L18P5p0CS,
              'ST13':CW120L18P5p0CS,
              ## S circles
              'ST6':CW120L10P5p0CS,
              'ST7':CW120L10P5p0CS,
              'ST8':CW120L10P5p0CS,
              'ST9':CW120L10P5p0CS,
              ## S circles    
              'ST2':CW120L5P5p0CS,
              'ST3':CW120L5P5p0CS,
              'ST4':CW120L5P5p0CS,
              'ST5':CW120L5P5p0CS,
              ## N none
              'NT1':None,
              ## N standard
              'NT2':RW80L80P5p0CS,
              'NT3':RW80L80P5p0CS,
              'NT4':RW80L80P5p0CS,
              'NT5':RW80L80P5p0CS,
              ## N sTandard
              'NT6':RW80L40P5p0CS,                  
              'NT7':RW80L40P5p0CS,
              'NT8':RW80L40P5p0CS,
              'NT9':RW80L40P5p0CS,
              ## N standard
              'NT10':RW80L20P5p0CS,
              'NT11':RW80L20P5p0CS,
              'NT12':RW80L20P5p0CS,
              'NT13':RW80L20P5p0CS,
              ## N standard
              'NT14':RW80L10P5p0CS,
              'NT15':RW80L10P5p0CS,
              'NT16':RW80L10P5p0CS,
              'NT17':RW80L10P5p0CS, 
              ## N standard
              'NT18':RW80L5P5p0CS,
              'NT19':RW80L5P5p0CS,
              'NT20':RW80L5P5p0CS,
              'NT21':RW80L5P5p0CS,
              'NT22':None,
              ## E None
              'ET1':None,              
              'ET22':None,
              ## E Finger 2
              'ET6':RW120L5P5p0CS,
              'ET11':RW120L5P5p0CS,
              'ET12':RW120L5P5p0CS,
              'ET17':RW120L5P5p0CS,
                
              ## E Finger 2
              'ET2':RW120L120P5p0CS,
              'ET7':RW120L120P5p0CS,
              'ET16':RW120L120P5p0CS,
              'ET21':RW120L120P5p0CS,
              ## E Finger 2
              'ET3':RW120L60P5p0CS,
              'ET8':RW120L60P5p0CS,
              'ET15':RW120L60P5p0CS,
              'ET20':RW120L60P5p0CS,
              ## E Finger 2
              'ET4':RW120L20P5p0CS,
              'ET9':RW120L20P5p0CS,
              'ET14':RW120L20P5p0CS,
              'ET19':RW120L20P5p0CS,
              ## E Finger 2
              'ET5':RW120L10P5p0CS,
              'ET10':RW120L10P5p0CS,
              'ET13':RW120L10P5p0CS,
              'ET18':RW120L10P5p0CS,
              ## O None
              'OT1':None,   
              'OT22':None,                      
              ## O Finger 3.5
              'OT5':CW80L16P5p0CS,
              'OT10':CW80L16P5p0CS,
              'OT13':CW80L16P5p0CS,
              'OT18':CW80L16P5p0CS,                
              ## O Finger 3.5
              'OT4':CW80L12P5p0CS,
              'OT9':CW80L12P5p0CS,
              'OT14':CW80L12P5p0CS,
              'OT19':CW80L12P5p0CS,
              ## O Finger 3.5
              'OT3':CW80L8P5p0CS,
              'OT8':CW80L8P5p0CS,
              'OT15':CW80L8P5p0CS,
              'OT20':CW80L8P5p0CS,
              ## O Finger 3.5
              'OT2':CW80L5P5p0CS,
              'OT7':CW80L5P5p0CS,
              'OT16':CW80L5P5p0CS,
              'OT21':CW80L5P5p0CS,
              ## O Finger 3.5
              'OT12':CW80L20P5p0CS,
              'OT17':CW80L20P5p0CS,
              'OT6':CW80L20P5p0CS,
              'OT11':CW80L20P5p0CS,                
              }
