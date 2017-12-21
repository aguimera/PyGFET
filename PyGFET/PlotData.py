# -*- coding: utf-8 -*-
"""
@author: Anton Guimerà
@version: 0.1b

Revsion history

- 0.1b -- First version

TODO implement graph for selected channels

"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.colors as colors
import matplotlib.cm as cmx
import NoiseModel as noise
from scipy import interpolate
import datetime 
from itertools import cycle
#from DaqInterface import *

##############################################################################
##### TODO comment this function with all options
###############################################################################
def PlotAC(Dev,Axes,PltUd0=None,**kwargs):

    LiTyp = cycle(["-","--","-.",":"])
    
    legend = False
    Clear = True
    channels = sorted(Dev)

    iVgs = None  
    iVds = None

    ColorOnVgs = False
    ColorOnChannels = True
    PSDalpha = 0.5
    ColorMap = cmx.jet
    
    for name, value in kwargs.items():       
        if name=='legend':
            legend = value
        if name=='channels':
            channels = value
        if name=='Clear':
            Clear = value
        if name=='iVgs':
            iVgs = (value,) 
        if name=='iVds':
            iVds = (value,)
        if name=='ColorOn':
            if value=='Channels':
                ColorOnChannels=True
                ColorOnVgs=False
            if value=='Vgs':
                ColorOnVgs=True
                ColorOnChannels=False
        if name=='ColorMap':
            ColorMap=value
        if name=='PSDalpha':
            PSDalpha=value
    
    if Clear:
        for ax in Axes:
            while Axes[ax].lines:
                Axes[ax].lines[0].remove()

    if ColorOnChannels:
        cmap = cmx.ScalarMappable(colors.Normalize(vmin=0, vmax=len(Dev)),
                              ColorMap)

    for ich,Ch in enumerate(channels):
        ch = Dev[Ch]
        mark = '-'        
        
        if iVds:
            ivds = iVds            
        else:
            ivds = range(len(ch['Vds']))        
        SLvds = ['Vd{}'.format(v) for v in ivds]    

                
        if iVgs:
            ivgs = iVgs
        else:
            ivgs = range(len(ch['Vgs']))
            
        for svds,ivd in zip(SLvds,ivds):
            if PltUd0:
                Vgs = Dev[Ch]['Vgs']-Dev[Ch]['Ud0'][ivd] ## +Dev[Ch]['Ud0'][ivd] ### search min
            else:
                Vgs = Dev[Ch]['Vgs']
                
            if ColorOnVgs:
                mark = Lt[ivd]
                cmap = cmx.ScalarMappable(
                                          colors.Normalize(vmin=np.min(ch['Vgs']), 
                                                           vmax=np.max(ch['Vgs'])),
                                          ColorMap)
            
            for ivg in ivgs:
                lb = Ch                
                if ColorOnChannels:                
                        color = cmap.to_rgba(ich)
                if ColorOnVgs:
                        color = cmap.to_rgba(ch['Vgs'][ivg])
                        lb = '{} Vgs {} Vds {}'.format(Ch,
                                                       round(ch['Vgs'][ivg],2),
                                                       ch['Vds'][ivd])
                        
                if 'IsOK' in ch:
                     if not ch['IsOK']:
                         mark='x'
        
                if 'GmMag' in Axes:
                    Axes['GmMag'].semilogx(ch['Fgm'],
                                np.abs(ch['gm'][svds][ivg,:]),
                                mark, color=color, label=lb)
                
                if 'GmPh' in Axes:
                    Axes['GmPh'].semilogx(ch['Fgm'],
                                np.angle(ch['gm'][svds][ivg,:])*180/np.pi,
                                mark, color=color,label=lb)

                if 'NoA' in Axes:
                    Axes['NoA'].semilogy(Vgs,
                                         ch['NoA'][:,ivd],
                                            mark, color=color, label=lb)
                if 'NoB' in Axes:
                    Axes['NoB'].plot(Vgs,
                                     ch['NoB'][:,ivd],
                                      mark, color=color)        

                if 'PSD' in Axes:
                    Axes['PSD'].loglog(ch['Fpsd'],
                                ch['PSD'][svds][ivg,:],
                                mark, color=color,alpha = PSDalpha, label=lb)
                    
                if 'PSD_V' in Axes:
                    Axes['PSD_V'].loglog(ch['Fpsd'],
                                ch['PSD'][svds][ivg,:],
                                mark, color=color,alpha = PSDalpha, label=lb)
        
                    if 'NoA' in ch:
                        Fit = noise.Fnoise(ch['Fpsd'][1:],
                                       ch['NoA'][ivg,ivd],
                                       ch['NoB'][ivg,ivd]),
        
                        Axes['PSD'].loglog(ch['Fpsd'][1:], Fit[0], '--',
                                            color=color) ### TODO not understand fit[0]

    if legend:
        Axes['PSD'].legend(fontsize='x-small', framealpha = 0.2)

###############################################################################
#####
###############################################################################
def PlotDC(Dev,Axes,iVds=None, legend=False, PlotNOk=True, Clear=True,PltUd0=None):
    lT = ('-','-v','-.','-.v','--','--v','-.v',':v')

    if Clear:
        for ax in Axes:
            while Axes[ax].lines:
                Axes[ax].lines[0].remove()        
            
    cmap = cmx.ScalarMappable(colors.Normalize(vmin=0, vmax=len(Dev)),cmx.jet)

    for ich,Ch in enumerate(sorted(Dev)):
        ch=Dev[Ch]
        
        if Ch=='Gate':
            if 'Ig' in Axes:
                Axes['Ig'].plot(ch['Vgs'],ch['Ig'],'ro')
                continue
        
        color = cmap.to_rgba(ich)
        
        if iVds:
            vdind = iVds
        else:
            vdind = range(len(ch['Vds']))
        
        for ivd in vdind:
            if PltUd0:
                Vgs = Dev[Ch]['Vgs']-Dev[Ch]['Ud0'][ivd] ## +Dev[Ch]['Ud0'][ivd] ### search min
            else:
                Vgs = Dev[Ch]['Vgs']
                
            Vds = ch['Vds'][ivd]
            lab = '{} {}'.format(Ch,Vds)
            
            Rds = Vds/ch['Ids'][:,ivd]
            mark = lT[ivd]

            if 'IsOK' in ch:
                if not ch['IsOK']:
                    mark='x'
                    if not PlotNOk: continue         

            if 'Rds' in Axes:
                Axes['Rds'].plot(Vgs,
                             Rds,
                             mark, color=color)           
            
            if 'Ids' in Axes:                        
                Axes['Ids'].plot(Vgs,
                             ch['Ids'][:,ivd],
                             mark, color=color)           
                if 'IdsPoly' in ch:
                    Axes['Ids'].plot(Vgs,
                             np.polyval(ch['IdsPoly'][:,ivd],ch['Vgs']),
                             'k-',alpha=0.3) 
            
            if 'Gm' in Axes:
                if 'GMPoly' in ch:                    
                    gm = np.polyval(ch['GMPoly'][:,ivd],ch['Vgs'])
                    Axes['Gm'].plot(Vgs,
                                gm,
                                mark, color=color, label=lab)
                else:
                    gm = np.diff(ch['Ids'][:,ivd])/np.diff(ch['Vgs'])                    
                    Axes['Gm'].plot(Vgs[1:],
                                gm,
                                mark, color=color, label=lab)

        if legend:
            Axes['Gm'].legend(fontsize='xx-small',
                            ncol=8, framealpha = 0.2, loc=0)


###############################################################################
#####
###############################################################################
def PlotCicle(Dev,Axes,iVds=0):
    
    GMax = np.ones((len(Dev)))*np.NaN;
    Ud = np.ones((len(Dev)))*np.NaN;
    Imin = np.ones((len(Dev)))*np.NaN;
    Cicle = range((len(Dev)))
    
    
    for ich,Ch in enumerate(sorted(Dev)):
        ch=Dev[Ch]      
        GMax[ich]=ch['GMax'][iVds]
        Ud[ich]=ch['Ud'][iVds]
        Imin[ich]=ch['Imin'][iVds]
        if 'Name' in ch:
            Label = ch['Name']
        else:
            Label = ch['ChName']
    
    if 'GMax' in Axes:        
        Axes['GMax'].plot(Cicle,GMax,label=Label)
        
    if 'Ud' in Axes:                    
        Axes['Ud'].plot(Cicle,Ud,label=Label)

    if 'Imin' in Axes:                    
        Axes['Imin'].plot(Cicle,Imin,label=Label)            
    
    Axes['Ud'].legend()
    
###############################################################################
#####
###############################################################################
def PlotACDCGm(DevAC, Ch, Axes, Clear = True,PltUd0=False):
    
    if Clear:
        for ax in Axes:
            while Axes[ax].lines:
                Axes[ax].lines[0].remove()
    
    VgsMin = np.min(DevAC[Ch]['Vgs'])
    VgsMax = np.max(DevAC[Ch]['Vgs'])
    
    lT = ('-','-v','-.','-.v','--','--v','-.v',':v')
    cmap = cmx.ScalarMappable(colors.Normalize(vmin=VgsMin, vmax=VgsMax),cmx.jet)
        
    for iVds,Vds in enumerate(DevAC[Ch]['Vds']):        
       
        sVd = 'Vd{}'.format(iVds)
        
        if PltUd0:
            VgsUd = DevAC[Ch]['Vgs']-DevAC[Ch]['Ud0'][iVds] ## +Dev[Ch]['Ud0'][ivd] ### search min
        else:
            VgsUd = DevAC[Ch]['Vgs']       
        
        for iVgs,Vgs in enumerate(VgsUd):                                  
                
            color = cmap.to_rgba(Vgs)
            
            if 'GmMag' in Axes:
                Axes['GmMag'].semilogx(DevAC[Ch]['Fgm'],
                            np.abs(DevAC[Ch]['gm'][sVd][iVgs,:]),
                            lT[iVds],color=color)
            
            if 'GmPh' in Axes:
                Axes['GmPh'].semilogx(DevAC[Ch]['Fgm'],
                            np.angle(DevAC[Ch]['gm'][sVd][iVgs,:])*180/np.pi,
                            lT[iVds],color=color)
                
            if 'Gm' in Axes:                     
                Axes['Gm'].plot(Vgs,
                                np.abs(np.polyval(DevAC[Ch]['GMPoly'][:,iVds],DevAC[Ch]['Vgs'][iVgs])),
                                'ro',color=color)                   


###############################################################################
#####
###############################################################################
def PlotACNoise (Dev,Axes,iVds=None, legend=True, Clear=True, PltUd0=False):
    lT = ('-','-v','-.','-.v','--','--v','-.v',':v')

    if Clear:
        for ax in Axes:
            while Axes[ax].lines:
                Axes[ax].lines[0].remove()

    cmap = cmx.ScalarMappable(colors.Normalize(vmin=0, vmax=len(Dev)),cmx.jet)

    for ich,Ch in enumerate(sorted(Dev)):
        if not 'NoA' in Dev[Ch]: continue
    
        if iVds:
            vdind = iVds
        else:
            vdind = range(len(Dev[Ch]['Vds']))
        
        color = cmap.to_rgba(ich)
        for ivd in vdind:
            mark = lT[ivd]
            lab = '{} {}'.format(Ch,Dev[Ch]['Vds'][ivd])             
            if PltUd0:
                Vgs = Dev[Ch]['Vgs']-Dev[Ch]['Ud0'][ivd] ## +Dev[Ch]['Ud0'][ivd] ### search min
            else:
                Vgs = Dev[Ch]['Vgs']
            
            if 'NoA' in Axes:
                Axes['NoA'].semilogy(Vgs,
                                     Dev[Ch]['NoA'][:,ivd],
                                     mark, color=color, label=lab)
            if 'NoB' in Axes:
                Axes['NoB'].plot(Vgs,
                                 Dev[Ch]['NoB'][:,ivd],
                                 '--', color=color)
            if 'FitErrA' in Axes:
                Axes['FitErrA'].semilogy(Vgs,
                                     Dev[Ch]['FitErrA'][:,ivd],
                                     mark, color=color, label=lab)
            if 'FitErrB' in Axes:
                Axes['FitErrB'].semilogy(Vgs,
                                     Dev[Ch]['FitErrB'][:,ivd],
                                     mark, color=color, label=lab)
                
            if not 'Irms' in Dev[Ch]: continue
            if 'Irms' in Axes:
                Axes['Irms'].plot(Vgs,
                                     Dev[Ch]['Irms'][:,ivd],
                                     mark, color=color, label=lab)
            if 'Vrms' in Axes:
                if 'GMPoly' in Dev[Ch]:                    
                    gm = np.polyval(Dev[Ch]['GMPoly'][:,ivd],Dev[Ch]['Vgs'])
                    Axes['Vrms'].semilogy(Vgs,
                                      Dev[Ch]['Irms'][:,ivd]/np.abs(gm),
                                      mark, color=color, label=lab)

        if legend:
            Axes['NoA'].legend(fontsize='xx-small',
                            ncol=4, framealpha = 0.2, loc=0)

###############################################################################
##### Figure with AC and DC axis in one figure
###############################################################################
def CreateACDCLiveFigure(Size=(15,10)):
    '''
    return Fig,AxDC,AxAC
    '''

    Fig = plt.figure(figsize=Size)
    gs = gridspec.GridSpec(2,4)

    ax = Fig.add_subplot(gs[0,0])
    Axs ={'Ids':ax,
         'Ig':ax.twinx(),
         'Rds':Fig.add_subplot(gs[0,1]),
         'Gm':Fig.add_subplot(gs[1,:2]),
         'GmMag':Fig.add_subplot(gs[0,2]),
         'GmPh':Fig.add_subplot(gs[1,2]),
         'PSD':Fig.add_subplot(gs[:,3])}

    SetAxesLabels(Axs)

    Fig.tight_layout(pad=0.2, h_pad=0.2, w_pad=0.2)

    return Fig,Axs

###############################################################################
#####
###############################################################################
def CreateACFigure(Size=(15,10)):
    '''
    return Fig,Axs
    '''
    Fig = plt.figure(figsize=Size)
    gs = gridspec.GridSpec(2,2)

    Axs = {'GmMag':Fig.add_subplot(gs[0,0]),
           'GmPh':Fig.add_subplot(gs[1,0]),
           'PSD':Fig.add_subplot(gs[:,1])}

    SetAxesLabels(Axs)

    Fig.tight_layout()
   
    return Fig,Axs

###############################################################################
#####
###############################################################################
def CreateDCFigure(Size=(15,10)):

    Fig = plt.figure(figsize=Size)
    gs = gridspec.GridSpec(2,2)
    ax = Fig.add_subplot(gs[0,0])

    Axs={'Ids':ax,
         'Ig':ax.twinx(),
         'Rds':Fig.add_subplot(gs[0,1]),
         'Gm':Fig.add_subplot(gs[1,:])}

    SetAxesLabels(Axs)

    Fig.tight_layout()
    return Fig,Axs

###############################################################################
#####
###############################################################################
def CreateACNoiseFigure(Size=(15,10)):

    Fig = plt.figure(figsize=Size)
    gs = gridspec.GridSpec(2,2)

    ax1 = Fig.add_subplot(gs[1,1])
    ax2 = Fig.add_subplot(gs[1,0])
    Axs={'NoA':Fig.add_subplot(gs[0,0]),
         'NoB':Fig.add_subplot(gs[0,1]),
         'FitErrA':ax1,
         'FitErrB':ax1.twinx(),
         'Irms':ax2,
         'Vrms':ax2.twinx()}
    SetAxesLabels(Axs)

    Fig.tight_layout()

    return Fig,Axs


###############################################################################
#####
###############################################################################
def CreateACDCGmFigure(Size=(15,10)):

    Fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=Size,
                                                sharex='col',sharey='row')
    
    Axs={'Gm':ax1,
         'DCPh':ax3,
         'GmMag':ax2,
         'GmPh':ax4}
         
    SetAxesLabels(Axs)

    Fig.tight_layout()

    return Fig,Axs


    
###############################################################################
#####
###############################################################################
def CreateDCCicleFigure(Size=(15,10)):

    Fig = plt.figure(figsize=Size)
    gs = gridspec.GridSpec(2,2)
    ax = Fig.add_subplot(gs[0,0])

    Axs={'Imin':ax,
         'GMax':Fig.add_subplot(gs[0,1]),
         'Ud':Fig.add_subplot(gs[1,:])}

    SetAxesLabels(Axs)
    
    Fig.tight_layout()
    return Fig,Axs  

    
###############################################################################
#####
###############################################################################
def CreateTestNoiseFigure(Size=(15,10)):

    '''
    return Fig,Axs
    '''
    Fig = plt.figure(figsize=Size)
    gs = gridspec.GridSpec(2,2)

    Axs = {'GmMag':Fig.add_subplot(gs[0,0]),
           'GmPh':Fig.add_subplot(gs[1,0]),
           'PSD':Fig.add_subplot(gs[0,1]),
           'PSD_V':Fig.add_subplot(gs[1,1])}

    SetAxesLabels(Axs)

    Fig.tight_layout()
   
    return Fig,Axs


###############################################################################
#####
###############################################################################
def CreateTestDCPCBFigure(Size=(15,10)):

    '''
    return Fig,Axs
    '''
    Fig = plt.figure(figsize=Size)
    gs = gridspec.GridSpec(2,2)

    Axs = {'Ids':Fig.add_subplot(gs[0,0]),
           'Gm':Fig.add_subplot(gs[1,0]),
           'GmMag':Fig.add_subplot(gs[0,1]),
           'GmPh':Fig.add_subplot(gs[1,1])}

    SetAxesLabels(Axs)

    Fig.tight_layout()
   
    return Fig,Axs     
    
###############################################################################
#####
###############################################################################
def SetAxesLabels(Axs, fontsize='medium', labelsize=5, 
                  scilimits = (-2,2), RdsLim=(200,1e4)):
    
    if 'Ids' in Axs:
        Axs['Ids'].set_ylabel('Ids[A]',fontsize=fontsize)
        Axs['Ids'].set_xlabel('Vgs[V]',fontsize=fontsize)
        Axs['Ids'].grid()
        Axs['Ids'].tick_params(axis='both', which='Both', labelsize=labelsize)
        Axs['Ids'].ticklabel_format(axis='y', style='sci', scilimits=scilimits)
        
    if 'Ig' in Axs:
        Axs['Ig'].set_ylabel('Ig[A]',fontsize=fontsize)
        Axs['Ig'].tick_params(axis='both', which='Both', labelsize=labelsize)
        Axs['Ig'].ticklabel_format(axis='y', style='sci', scilimits=scilimits)

 
    if 'Rds' in Axs:
        Axs['Rds'].set_ylabel(r'$RDS [\Omega]$',fontsize=fontsize);
        Axs['Rds'].set_xlabel('Vgs[V]',fontsize=fontsize)
        Axs['Rds'].grid()
        Axs['Rds'].tick_params(axis='both', which='Both', labelsize=labelsize)
        Axs['Rds'].ticklabel_format(axis='y', style='sci', scilimits=scilimits)
        if RdsLim:
            Axs['Rds'].set_ylim(RdsLim[0],RdsLim[1])

    if 'Gm' in Axs:
        Axs['Gm'].set_ylabel('Gm[S]',fontsize=fontsize)
        Axs['Gm'].set_xlabel('Vgs[V]',fontsize=fontsize)
        Axs['Gm'].grid()
        Axs['Gm'].tick_params(axis='both', which='Both', labelsize=labelsize)
        Axs['Gm'].ticklabel_format(axis='y', style='sci', scilimits=scilimits)

    if 'GmMag' in Axs:
        Axs['GmMag'].set_ylabel('Gm[S]',fontsize=fontsize)
        Axs['GmMag'].set_xlabel('Frequency [Hz]',fontsize=fontsize)
        Axs['GmMag'].grid()
        Axs['GmMag'].tick_params(axis='both', which='Both', labelsize=labelsize)
        Axs['GmMag'].ticklabel_format(axis='y', style='sci', scilimits=scilimits)

    if 'GmPh' in Axs:
        Axs['GmPh'].set_ylabel('Phase[0]',fontsize=fontsize)
        Axs['GmPh'].set_xlabel('Frequency [Hz]',fontsize=fontsize)
        Axs['GmPh'].grid()
        Axs['GmPh'].tick_params(axis='both', which='Both', labelsize=labelsize)
        Axs['GmPh'].ticklabel_format(axis='y', style='sci', scilimits=scilimits)
        
    if 'PSD' in Axs:
        Axs['PSD'].set_ylabel('PSD [A2/Hz]',fontsize=fontsize)
        Axs['PSD'].set_xlabel('Frequency [Hz]',fontsize=fontsize)
        Axs['PSD'].grid()
        Axs['PSD'].tick_params(axis='both', which='Both', labelsize=labelsize)
        Axs['PSD'].ticklabel_format(axis='y', style='sci', scilimits=scilimits)
    
    if 'NoA' in Axs:    
        Axs['NoA'].set_ylabel('a [A^2]',fontsize=fontsize)
        Axs['NoA'].set_xlabel('Vgs [V]',fontsize=fontsize)      
        Axs['NoA'].grid()
        Axs['NoA'].tick_params(axis='both', which='Both', labelsize=labelsize)
        Axs['NoA'].ticklabel_format(axis='y', style='sci', scilimits=scilimits)
        
    if 'NoB' in Axs:    
        Axs['NoB'].set_ylabel('b []',fontsize=fontsize)
        Axs['NoB'].set_xlabel('Vgs [V]',fontsize=fontsize)    
        Axs['NoB'].grid()
        Axs['NoB'].tick_params(axis='both', which='Both', labelsize=labelsize)
        Axs['NoB'].ticklabel_format(axis='y', style='sci', scilimits=scilimits)

    if 'Irms' in Axs:
        Axs['Irms'].set_ylabel('Irms [Arms]',fontsize=fontsize)
        Axs['Irms'].set_xlabel('Vgs [V]',fontsize=fontsize)    
        Axs['Irms'].grid()
        Axs['Irms'].tick_params(axis='both', which='Both', labelsize=labelsize)
        Axs['Irms'].ticklabel_format(axis='y', style='sci', scilimits=scilimits)
    
    if 'Vrms' in Axs:
        Axs['Vrms'].set_ylabel('Vrms [Vrms]',fontsize=fontsize)
        Axs['Vrms'].set_xlabel('Vgs [V]',fontsize=fontsize)    
        Axs['Vrms'].grid()
        Axs['Vrms'].tick_params(axis='both', which='Both', labelsize=labelsize)
        Axs['Vrms'].ticklabel_format(axis='y', style='sci', scilimits=scilimits)

    if 'FitErrA' in Axs:
        Axs['FitErrA'].set_ylabel('a fit error',fontsize=fontsize)
        Axs['FitErrA'].set_xlabel('Vgs [V]',fontsize=fontsize)    
        Axs['FitErrA'].grid()
        Axs['FitErrA'].tick_params(axis='both', which='Both', labelsize=labelsize)
        Axs['FitErrA'].ticklabel_format(axis='y', style='sci', scilimits=scilimits)

    if 'FitErrB' in Axs:
        Axs['FitErrB'].set_ylabel('b fit error',fontsize=fontsize)
        Axs['FitErrB'].set_xlabel('Vgs [V]',fontsize=fontsize)    
        Axs['FitErrB'].grid()
        Axs['FitErrB'].tick_params(axis='both', which='Both', labelsize=labelsize)
        Axs['FitErrB'].ticklabel_format(axis='y', style='sci', scilimits=scilimits)

    if 'GMax' in Axs:
        Axs['GMax'].set_ylabel('Max. transconductance [S]',fontsize=fontsize)
        Axs['GMax'].set_xlabel('Cicle',fontsize=fontsize)
        Axs['GMax'].grid()
        Axs['GMax'].tick_params(axis='both', which='Both', labelsize=labelsize)
        Axs['GMax'].ticklabel_format(axis='y', style='sci', scilimits=scilimits)
    
    if 'Ud' in Axs:
        Axs['Ud'].set_ylabel('Dirac Point [V]',fontsize=fontsize)
        Axs['Ud'].set_xlabel('Cicle',fontsize=fontsize)
        Axs['Ud'].grid()
        Axs['Ud'].tick_params(axis='both', which='Both', labelsize=labelsize)
        Axs['Ud'].ticklabel_format(axis='y', style='sci', scilimits=scilimits)
    
    if 'Imin' in Axs:
        Axs['Imin'].set_ylabel('Min. Current [A]',fontsize=fontsize)
        Axs['Imin'].set_xlabel('Cicle',fontsize=fontsize)
        Axs['Imin'].grid()
        Axs['Imin'].tick_params(axis='both', which='Both', labelsize=labelsize)
        Axs['Imin'].ticklabel_format(axis='y', style='sci', scilimits=scilimits)

    if 'PSD_V' in Axs:
        Axs['PSD_V'].set_ylabel('PSD [V2/Hz]',fontsize=fontsize)
        Axs['PSD_V'].set_xlabel('Frequency [Hz]',fontsize=fontsize)
        Axs['PSD_V'].grid()
        Axs['PSD_V'].tick_params(axis='both', which='Both', labelsize=labelsize)
#        Axs['PSD_V'].tick_params(axis='y', which='both', labelleft='off', labelright='on')
        Axs['PSD_V'].ticklabel_format(axis='y', style='sci', scilimits=scilimits)
               
    if 'Vout' in Axs:
        Axs['Vout'].set_ylabel('Vout[V]',fontsize=fontsize)
        Axs['Vout'].tick_params(axis='both', which='Both', labelsize=labelsize)
        Axs['Vout'].ticklabel_format(axis='y', style='sci', scilimits=scilimits)
    
    
