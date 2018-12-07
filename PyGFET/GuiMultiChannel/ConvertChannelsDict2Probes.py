# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 14:55:18 2018

@author: Javier
"""

import deepdish as dd


Path = 'Y:\\PyFET'
FileName = 'B11870W8-Q3Q4-AC-Cy1.h5'
#FileName = 'B11870W8-Q3Q4-Cy0.h5'
File = Path + '\\' + FileName
Name1='B11870W8-Q3-AC-Cy1.h5'
Name2='B11870W8-Q4-AC-Cy1.h5'

Dict = dd.io.load(File)

DevDCVals1 = {}
DevDCVals2 = {}
DevACVals1 = {}
DevACVals2 = {}

if type(Dict) is tuple: 
    DCdict=Dict[0]
    for i, n in zip(sorted(DCdict.keys()), range(len(DCdict))):
        if i != 'Gate':
    #        print i, n
            number = int(i.split('Ch')[1])
            if number < 17:
                DevDCVals1[i] = DCdict[i]
            else:
                NewNum = number - 16
                if NewNum < 10:
                    NewName = 'Ch0' + str(NewNum)
                else:
                    NewName = 'Ch' + str(NewNum)
                
                DevDCVals2[i] = DCdict[i]
                DevDCVals2[i]['Name'] = NewName
                DevDCVals2[i]['ChName'] = NewName
                ##
                DevDCVals2[NewName] = DevDCVals2.pop(i)
        else:
            DevDCVals2[i] = DCdict[i]
            DevDCVals1[i] = DCdict[i]    
            
    ACdict=Dict[1]
    for i, n in zip(sorted(ACdict.keys()), range(len(ACdict))):
        if i != 'Gate':
    #        print i, n
            number = int(i.split('Ch')[1])
            if number < 17:
                DevACVals1[i] = ACdict[i]
            else:
                NewNum = number - 16
                if NewNum < 10:
                    NewName = 'Ch0' + str(NewNum)
                else:
                    NewName = 'Ch' + str(NewNum)
                
                DevACVals2[i] = ACdict[i]
                DevACVals2[i]['Name'] = NewName
                DevACVals2[i]['ChName'] = NewName
                ##
                DevACVals2[NewName] = DevACVals2.pop(i)
#        else:
#            DevACVals2[i] = ACdict[i]
#            DevACVals1[i] = ACdict[i]
    
    
else:  
    for i, n in zip(sorted(Dict.keys()), range(len(Dict))):
        if i != 'Gate':
    #        print i, n
            number = int(i.split('Ch')[1])
            if number < 17:
                DevDCVals1[i] = Dict[i]
            else:
                NewNum = number - 16
                if NewNum < 10:
                    NewName = 'Ch0' + str(NewNum)
                else:
                    NewName = 'Ch' + str(NewNum)
                
                DevDCVals2[i] = Dict[i]
                DevDCVals2[i]['Name'] = NewName
                DevDCVals2[i]['ChName'] = NewName
                ##
                DevDCVals2[NewName] = DevDCVals2.pop(i)
        else:
            DevDCVals2[i] = Dict[i]
            DevDCVals1[i] = Dict[i]


dd.io.save(Path+'\\' + Name1,(DevDCVals1,DevACVals1))
dd.io.save(Path+'\\' + Name2,(DevDCVals2,DevACVals2))

        