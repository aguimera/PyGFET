#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 10:32:57 2018

@author: aguimera
"""

import PyGFET.DBXlsReport as XlsRep

Name = 'B10631W3'
Conditions = {'Wafers.name=': (Name, )}

XlsRept = XlsRep.GenXlsReport(Name+'.xlsx',
                              Conditions)

XlsRept.GenFullReport()
XlsRept.close()


