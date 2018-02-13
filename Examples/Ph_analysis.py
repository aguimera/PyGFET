# -*- coding: utf-8 -*-
"""
Created on Mon Jan 08 17:23:06 2018

@author: eduard
"""

import matplotlib.colors as mpcolors
import matplotlib.cm as cmx
import PyGFET.DBAnalyze as Dban
#==============================================================================
# from PyFET.ExportTools import SaveOpenSigures
#==============================================================================
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import statsmodels.api as sm

import datetime
from matplotlib.backends.backend_pdf import PdfPages

#%% Find r in DB

plt.close('all')
plt.ion()

CharTable = 'DCcharacts'
DeviceNames = ('B10803W16-Xip2E',)

Conditions = {'Devices.Name=': DeviceNames,
              'CharTable.IsOK>': (0, ),
              'CharTable.FuncStep=': ('SScovCOOH',),
              'CharTable.Ph<>': (10.0,),
              'CharTable.Ph>': (1,)}

TrtsList = Dban.FindCommonParametersValues(Table=CharTable,
                                           Parameter='Trts.Name',
                                           Conditions=Conditions)


PhList = Dban.FindCommonParametersValues(Table=CharTable,
                                           Parameter='CharTable.Ph',
                                           Conditions=Conditions)

#IonStrenList = Dban.FindCommonParametersValues(Table=CharTable,
#                                           Parameter='CharTable.IonStrength',
#                                           Conditions=Conditions)

Groups = {}
# Fixed Conditions
CondBase = {}
CondBase['Table'] = CharTable
CondBase['Last'] = False
CondBase['Conditions'] = Conditions


for TrtN in sorted(TrtsList):
    Cgr = CondBase.copy()
    Cond = CondBase['Conditions'].copy()
    Cond.update({'Trts.Name=': (TrtN, )})
    Cgr['Conditions'] = Cond
    Groups[TrtN] = Cgr

    dat, t= Dban.GetFromDB(Conditions=Cond, Table=CharTable, Last=False)
    
    #Dban.MultipleSearch(Func=Dban.PlotXYVars,
    #                    Groups=Groups,
    #                    Xvar='Ph',
    #                    Yvar='Ud0',
    #                    Vgs=None,
    #                    Vds=None)
    #
    #plt.show()     
    #
    ##%%
    #
    #plt.close('all')
    
    Dats = dat[TrtN]
    
#    x = np.ones([len(Dats)])
    y = np.ones([len(Dats)])
    z = np.ones([len(Dats)])
    for i, d in enumerate(Dats):
        y[i] = d.GetPh()
#        x[i] = d.GetIonSt()
        z[i] = d.GetUd0()
    
    plt.figure()
#    plt.tricontourf(x,y,z,100,cmap='seismic')
#    plt.tricontourf(x,y,z,100,cmap='seismic')
    plt.plot(x,y,'ko')
    plt.colorbar()
#
#==============================================================================
# CondBase2 = {}
# CondBase2['Table'] = CharTable
# CondBase2['Last'] = True
# CondBase2['Conditions'] = Conditions        
#==============================================================================

#==============================================================================
# plt.figure()
#==============================================================================



# Create the PdfPages object to which we will save the pages:
# The with statement makes sure that the PdfPages object is closed properly at
## the end of the block, even if an Exception occurs.
#with PdfPages('multipage_pdf.pdf') as pdf:
#
#
#
#    Res = np.array([])
#    R2 = np.array([])
#    cmap = cmx.ScalarMappable(mpcolors.Normalize(vmin=0, vmax=len(TrtsList)),
#                              cmx.jet)
#    for itrt, Trt in enumerate(sorted(TrtsList)):
#        color = cmap.to_rgba(itrt)
#        Conditions.update({'Trts.Name=': (Trt,)})
#        ValX = np.array([])
#        ValY = np.array([])
#        
#        
#        for c,Conc in enumerate(sorted(AnalyteConList)):
#            if c==0:
#                Groups = {}
#                
#                for Conc2 in sorted(AnalyteConList):
#                    Cgr = CondBase.copy()
#                    Cond = CondBase['Conditions'].copy()
#                    Cond.update({'CharTable.IonStrength=':(Conc2,)})
#                    Cgr['Conditions'] = Cond
#                    Groups[Conc2] = Cgr
#    #==============================================================================
#    #             f, (ax1, ax2) = plt.subplots(1, 2)          
#    #==============================================================================
#                Dban.MultipleSearch(Groups=Groups,
#                                    Xvar='Vgs',
#                                    Yvar='Ids',
#    #==============================================================================
#    #                                 Ax=ax1,
#    #                                 fig=f,
#    #==============================================================================
#                                    PlotOverlap=False,
#                                    Ud0Norm=Ud0Norm)
#                pdf.savefig()
#    #==============================================================================
#    #             f, (ax1, ax2) = plt.subplots(1, 2)
#    #==============================================================================
#    #==============================================================================
#    #             ax1.set_title(Trt)
#    #==============================================================================
#    
#    
#            Conditions.update({'CharTable.IonStrength=':(Conc,)})
#            
#            vy = Dban.MultipleSearchParam(Groups={'1': CondBase},
#                                            Plot=False,
#                                            Boxplot=False,
#                                            Param='Ud0').values()[0][0,0]
#            
#            ValY = np.hstack((ValY, vy)) if ValY.size else vy
#            ValX = np.hstack((ValX, Conc)) if ValX.size else np.array(Conc)
#        f,ax2=plt.subplots()
#        if Type=='IonStrength':
#            ax2.semilogx(ValX, ValY, 'o', color=color, label=Trt)
#            ax2.set_xlabel('IonStrength [M]')
#    #==============================================================================
#    #         ValX = np.log10(ValX)
#    #==============================================================================
#            X = sm.add_constant(np.log10(ValX))
#            
#        else:    
#            ax2.plot(ValX, ValY, 'o', color=color, label=Trt)    
#            X = sm.add_constant(ValX)
#            ax2.xlabel('pH')
#        
#        ax2.set_ylabel('CNP [V]')
#        res=sm.OLS(ValY, X).fit()
#        R2=np.vstack((R2,res.rsquared)) if R2.size else res.rsquared
#        prstd, iv_l, iv_u = wls_prediction_std(res)    
#        
#        ax2.plot(ValX, res.fittedvalues,'k--')
#        ax2.fill_between(ValX, iv_u, iv_l,
#                         color=color,
#                         linewidth=0.0,
#                         alpha=0.3)
#        Res = np.vstack((Res,res.params)) if Res.size else res.params
#        slope=res.params[1]*1000
#        ax2.legend(loc='best')
#    #==============================================================================
#    #     plt.title('slope= %.1f mV r2= %.1f' % slope % res.rsquared)
#    #==============================================================================
#        ax2.set_title('slope= {:.2f} mV'.format(slope) + ' r2={:.3f}'.format(res.rsquared))
#        pdf.savefig()
#    d = pdf.infodict()
#    d['Title'] = 'ph-ion sensibility analysis'
#    d['Author'] = u'Eduard Masvidal Codina'
#    d['Subject'] = 'ph-ion sensibility analysis'
##==============================================================================
##     d['Keywords'] = 'PdfPages multipage keywords author title subject'
##==============================================================================
##==============================================================================
##     d['CreationDate'] = datetime.datetime(2009, 11, 13)
##==============================================================================
#    d['CreationDate'] = datetime.datetime.today()
##==============================================================================
##     d['ModDate'] = datetime.datetime.today()
##==============================================================================
##==============================================================================
## plt.legend(loc='best')
##==============================================================================
#slope=[]
#slope=Res[:,1]
#slope=slope.reshape(len(slope),1)
#slope[R2<0.95]=np.nan
#R2[R2<0.95]=np.nan
#print np.nanmean(slope)
#print np.nanstd(slope)
#print np.nanmean(R2)
#    plt.plot(ValX,ValY)    