#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 13:33:19 2016

@author: aguimera
"""

import pylatex as pyl
import os
from PyFET.PlotData import *
from PyFET.AnalyzeData import *
import matplotlib.pyplot as plt


def AddImagePdf(plot, fig, width, TmpPath='./Graph/'):
   
    plt.figure(fig.number)   
    
    if os.name=='nt':      
        figname='{}{}{}'.format(TmpPath,fig.number,'.png')    
        plt.savefig(figname)    
        image_filename=os.path.abspath(figname)
        image_filename=image_filename.replace('\\','/')
        
        plot.add_image(image_filename, width=width)
    else:
        plot.add_plot(width=width) 


def GenReportPDF(DevDCVals,DevACVals,FileName):
  
   
    doc = pyl.Document()
    doc.packages.append(pyl.Package('geometry', options=['tmargin=1cm',
                                                     'lmargin=1cm']))
    
    FigDC,AxsDC = CreateDCFigure()
    
    PlotDC(DevDCVals,AxsDC,legend=True)
    
    with doc.create(pyl.Section('DC characterization')):
        with doc.create(pyl.Figure(position='htbp')) as plot:
            AddImagePdf(plot,FigDC,width='20cm')
            plot.add_caption(caption='DC characterization for all channels')

    if DevACVals:
        FigAC,AxsAC = CreateACFigure()
        FigGM,AxsGM = CreateACDCGmFigure()
    #PlotAc(DevACVals,Axs,channels=('Ch02','Ch04','Ch12'),legend=True,ColorOn='Vgs')

        with doc.create(pyl.Section('AC characterization')):
            FigNO,AxsNO = CreateACNoiseFigure()
            PlotACNoise (DevACVals,AxsNO,legend=True)
            with doc.create(pyl.Figure(position='htbp')) as plot:
                    AddImagePdf(plot,FigAC,width='20cm')
                    plot.add_caption(caption='Noise for all channels')
                                  
            for ch in sorted(DevACVals):               
                   #        with doc.create(pyl.Subsection(ch)): TODO fix the subsection generation
                    plt.figure(FigAC.number)
                    PlotAC(DevACVals,AxsAC,legend=True,channels=(ch,),ColorOn='Vgs')    
                    
                    with doc.create(pyl.Figure(position='htbp')) as plot:
                       AddImagePdf(plot,FigAC,width='20cm')
                       plot.add_caption(caption='AC characterization {}'.format(ch))
                       
                    plt.figure(FigGM.number)
                    PlotACDCGm(DevACVals,DevDCVals,ch,AxsGM)
                    with doc.create(pyl.Figure(position='htbp')) as plot:
                        AddImagePdf(plot,FigGM,width='20cm')
                        plot.add_caption(caption='AC vs DC characterization {}'.format(ch))
        
    doc.generate_pdf(FileName)
    
def GenDCCiclePDF(DevDCVals,FileName):
  
   
    doc = pyl.Document()
    doc.packages.append(pyl.Package('geometry', options=['tmargin=1cm',
                                                     'lmargin=1cm']))
    
    FigDC,AxsDC = CreateDCFigure()
    
    PlotDC(DevDCVals, AxsDC)
    
    with doc.create(pyl.Section('DC characterization')):
        with doc.create(pyl.Figure(position='htbp')) as plot:
            AddImagePdf(plot,FigDC,width='20cm')
            plot.add_caption(caption='DC characterization for all channels')

    CalcDCparams(DevDCVals)
    
    FigDCCicle,AxsDCCicle = CreateDCCicleFigure()
    
    PlotCicle(DevDCVals, AxsDCCicle)
    
    with doc.create(pyl.Section('Cicle Characterization')):
        with doc.create(pyl.Figure(position='htbp')) as plot:
            AddImagePdf(plot,FigDCCicle,width='20cm')
            plot.add_caption(caption='Cicle Characterization for all channels')
       
    
            
    doc.generate_pdf(FileName)