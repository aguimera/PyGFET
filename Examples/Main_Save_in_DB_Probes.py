# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 23:15:19 2016

@author: aguimera
"""
import PyGFET.DataStructures as PyData
import PyGFET.DBCore as Pydb
import PyGFET.AnalyzeData as PyAnalyze

import gc
import glob


#############################################################################
# Upload files
#############################################################################

MyDB = Pydb.PyFETdb(host='opter6.cnm.es',user='pyfet',passwd='p1-f3t17',db='pyFET')

#==============================================================================
# FileFilter = 'S:\Software\Python_DAQ\Andrea\python\Data\Probes\B8809O5-T7-ACDC_long.h5'
#==============================================================================
#FileFilter = './Data_Probes/*.h5'
#==============================================================================
# FileFilter = './Data/*.h5'
#==============================================================================
#FileFilter = '/media/aguimera/6B87-8F3F/python/Data/OBL23*'
#FileFilter = './DataDUTs/B9355O25-F3C3E-1-12-DC0-Cy*.h5'
#==============================================================================
# FileFilter = 'S:\Software\Python_DAQ\Andrea\python\Data\Probes\*'
#==============================================================================
#==============================================================================
# FileFilter = 'S:\Software\Python_DAQ\Andrea\python\Data\Probes\New folder\*.h5'
#==============================================================================
#==============================================================================
# FileFilter='S:\Software\Python_DAQ\Andrea\python\Data\Probes\B8809O5-T7\B8809O5-T7-ACDC_long.h5'
#==============================================================================
#==============================================================================
# FileFilter='S:\Software\Python_DAQ\Andrea\python\Data\Probes\Data-Selection\B8809O5-T11-DC-Cy0.h5'
#==============================================================================
#==============================================================================
# FileFilter='V:\PythonScripts\Examples\DataPenetrating\B9872W14-B4P*.h5'
#FileFilter='S:\Software\Python_DAQ\Andrea\python\Data\Corticals\B10631O24-BL-Dry\New folder\*.h5'
#FileFilter='S:\Users\Jessica\HallBar-TRTCharact\*.h5'
#==============================================================================
#FileFilter = 'S:\Software\Python_DAQ\Andrea\python\Data\Corticals\B10631W22-Al2O3-TopContact\*.h5'
#FileFilter='S:\Software\Python_DAQ\Andrea\python\Data\Intra-Cortical\B9872W4\B9872W4-P2-AC2-Cy0.h5'
FileFilter = 'C:\Data\Probes16trt\B10642W16-T6*.h5'
#FileFilter = 'C:\Data\Beforeuploading\*.h5
FileNames = glob.glob(FileFilter)

Fields = {}
Fields['User'] = 'JMA'

#TrtType = {'Name' : 'RW500L250P3CS',
#             'Length':250e-6, 
#             'Width':500e-6,
#             'Pass':3e-6,
#             'Contact':'Flat',
#             'Shape':'Rectangular'}
      
#TrtType = {'Name' : 'HW40L170P0CS',
#             'Length':170e-6, 
#             'Width':40e-6,
#             'Pass':0,
#             'Contact':'Flat',
#             'Shape':'Rectangular'}
#      
#TrtType = {'Name' : 'RW100L50P3CS',
#             'Length':50e-6, 
#             'Width':100e-6,
#             'Pass':3e-6,
#             'Contact':'Flat',
#             'Shape':'Rectangular'}
#==============================================================================
#==============================================================================
TrtType = {'Name' : 'RW80L30P3CS',
               'Length':30e-6, 
               'Width':80e-6,
               'Pass':3e-6,
               'Contact':'Flat',
               'Shape':'Rectangular'}
#==============================================================================
#==============================================================================

OptFields = {}
OptFields['Solution'] = '10mM PBS'    
OptFields['Comments'] = ''

         

GateFields={}

for ifile, filen in enumerate(FileNames):
    
    fName = filen.split('\\')[-1]
    fNameP = fName.split('-')
    ##### Naming rules B9355O15-T3-*****.h5
    #####              [0] Wafer                             
    #####                       [1] Device                             
    #####                           [] anything else
    
    print 'Load {} {} of {}'.format(fName, ifile, len(FileNames)) 
    
    Fields['Wafer'] = fNameP[0]
    Fields['Device'] = '{}-{}'.format(Fields['Wafer'],fNameP[1])
    
    print 'Device ', Fields['Device'] 

    ######## Load Data    
    DevDCVals,DevACVals = PyData.LoadDataFromFile(filen)        
    PyAnalyze.CheckIsOK (DevDCVals,DevACVals, RdsRange = [300,20e3])
    PyAnalyze.CalcGM (DevDCVals,DevACVals)

    if DevACVals:
        PyAnalyze.InterpolatePSD(DevACVals,Points=100)
        PyAnalyze.FitACNoise(DevACVals,Fmin=5, Fmax=1e3)
        PyAnalyze.CalcNoiseIrms(DevACVals)    

    
#    GenReportPDF (DevDCVals,DevACVals,'./Reports/{}'.format(filen.split('/')[-1]))
    
    OptFields['FileName'] = fName

    if 'Gate' in DevDCVals:
        GateFields['User'] = Fields['User']
        GateFields['Name'] = '{}-Gate'.format(Fields['Device'])
        GateFields['FileName'] = fName
        Fields['Gate_id'] = MyDB.InsertGateCharact(DevDCVals['Gate'], GateFields)
    else:
        Fields['Gate_id'] = None

    
    for ch in DevDCVals:        
        if ch=='Gate': continue
    
        Fields['Trt'] = '{}-{}'.format(Fields['Device'],                                       
                                        ch)   
        
        ##### Transistor Type definition          
 
        Fields['TrtType'] = TrtType['Name']
        print 'Trt ', Fields['Trt'], 'Type --> ', Fields['TrtType'] , ' L = ', TrtType['Length'], ' gid ', Fields['Gate_id'] 

        ###### Update DataBase        
        if DevACVals:            
            MyDB.InsertCharact(DCVals = DevDCVals[ch],
                           ACVals = DevACVals[ch],
                           Fields = Fields,
                           OptFields = OptFields,
                           TrtTypeFields = TrtType)
        else:
            MyDB.InsertCharact(DCVals = DevDCVals[ch],
                           ACVals = None,
                           Fields = Fields,
                           OptFields = OptFields,
                           TrtTypeFields = TrtType)
            

del MyDB
print 'Collect -->',gc.collect()
