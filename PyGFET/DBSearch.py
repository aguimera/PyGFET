#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 13:12:37 2018

@author: aguimera
"""

from PyGFET.DataClass import DataCharAC
import PyGFET.DBCore as PyFETdb
import numpy as np

def CheckConditionsCharTable(Conditions, Table):
    for k in Conditions.keys():
        if k.startswith('CharTable'):
            nk = k.replace('CharTable', Table)
            Conditions.update({nk: Conditions[k]})
            del(Conditions[k])
    return Conditions


def FindCommonValues(Parameter, Conditions, Table='ACcharacts', **kwargs):
    Conditions = CheckConditionsCharTable(Conditions, Table)

    if Parameter.startswith('CharTable'):
        Parameter = Parameter.replace('CharTable', Table)

    MyDb = PyFETdb.PyFETdb(host='opter6.cnm.es',
                           user='pyfet',
                           passwd='p1-f3t17',
                           db='pyFET')
#    MyDb = PyFETdb.PyFETdb()

    Output = (Parameter,)
    Res = MyDb.GetCharactInfo(Table=Table,
                              Conditions=Conditions,
                              Output=Output)

    del (MyDb)
    #  Generate a list of tupples with devices Names and comments
    Values = []
    for Re in Res:
        Values.append(Re[Parameter])

    return set(Values)


def GetFromDB(Conditions, Table='ACcharacts', Last=True, GetGate=True,
              OutilerFilter=None):
    Conditions = CheckConditionsCharTable(Conditions, Table)

    MyDb = PyFETdb.PyFETdb(host='opter6.cnm.es',
                           user='pyfet',
                           passwd='p1-f3t17',
                           db='pyFET')

#    MyDb = PyFETdb.PyFETdb()

    DataD, Trts = MyDb.GetData2(Conditions=Conditions,
                                Table=Table,
                                Last=Last,
                                GetGate=GetGate)

    del(MyDb)

    Data = {}
    for Trtn, Cys in DataD.iteritems():
        print Trtn
        Chars = []
        for Cyn, Dat in Cys.iteritems():
            Char = DataCharAC(Dat)
            Chars.append(Char)
        Data[Trtn] = Chars
    if OutilerFilter is None:
        return Data, list(Trts)

#   Find Outliers
    Vals = np.array([])
    for Trtn, Datas in Data.iteritems():
        for Dat in Datas:
            func = Dat.__getattribute__('Get' + OutilerFilter['Param'])
            Val = func(Vgs=OutilerFilter['Vgs'],
                       Vds=OutilerFilter['Vds'],
                       Ud0Norm=OutilerFilter['Ud0Norm'])
            if Val is not None:
                Vals = np.hstack((Vals, Val)) if Vals.size else Val

    p25 = np.percentile(Vals, 25)
    p75 = np.percentile(Vals, 75)
    lower = p25 - 1.5 * (p75 - p25)
    upper = p75 + 1.5 * (p75 - p25)

    Data = {}
    for Trtn, Cys in DataD.iteritems():
        Chars = []
        for Cyn, Dat in Cys.iteritems():
            Char = DataCharAC(Dat)
            func = Char.__getattribute__('Get' + OutilerFilter['Param'])
            Val = func(Vgs=OutilerFilter['Vgs'],
                       Vds=OutilerFilter['Vds'],
                       Ud0Norm=OutilerFilter['Ud0Norm'])

            if (Val <= lower or Val >= upper):
                print 'Outlier Removed ->', Val, Trtn, Cyn
            else:
                Chars.append(Char)
        Data[Trtn] = Chars

    return Data, list(Trts)


def UpdateCharTableField(Conditions, Value,
                         Table='ACcharacts', Field='Comments'):

    Conditions = CheckConditionsCharTable(Conditions, Table)

    MyDb = PyFETdb.PyFETdb(host='opter6.cnm.es',
                           user='pyfet',
                           passwd='p1-f3t17',
                           db='pyFET')

    out = '{}.id{}'.format(Table, Table)
    re = MyDb.GetCharactInfo(Table=Table,
                             Conditions=Conditions,
                             Output=(out, ))

    print re
    text = "Do you wan to update {} in {} for {} y/n ?".format(Field, Table, Value)
    inText = raw_input(text)
    if inText =='y':
        print 'Updated'
        field = '{}.{}'.format(Table, Field)
        fields = {field: Value}
        for r in re:
            condition = ('{}.id{}='.format(Table, Table), r.values()[0])
            MyDb.UpdateRow(Table=Table, Fields=fields, Condition=condition)
    else:
        print 'Cancelled'

    MyDb.db.commit()
