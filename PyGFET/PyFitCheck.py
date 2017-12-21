# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import sys
from PyQt5  import QtWidgets, uic
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMessageBox
import PyFET.PyFETdb as PyFETdb
import PyFET.PlotData as PyFETPlot
import matplotlib.pyplot as plt

qtCreatorFile = "./PyFET/FitCheck.ui" # Enter file here.

Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)

class AppFitCheck(QtWidgets.QMainWindow, Ui_MainWindow):    
    
    def __init__(self, ACData):
        QtWidgets.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        
        self.setWindowTitle('Fitting check')
        
        self.Data = ACData
        
        self.Axs={}
        self.fig, self.Axs['PSD'] = plt.subplots()        
        PyFETPlot.SetAxesLabels(self.Axs)
        
        self.FigNo, self.AxsNo = PyFETPlot.CreateACNoiseFigure()
        
        for TrtName in sorted(self.Data) :
            self.CmbTrts.addItem(TrtName)
        
        self.SpinVd.setMinimum(0)
        self.SpinVg.setMinimum(0)
        
        self.CmbChar.activated.connect(self.CharChange)
        self.CmbTrts.activated.connect(self.TrtChange)
        
        self.SpinVd.valueChanged.connect(self.VdChange)
        self.SpinVg.valueChanged.connect(self.VgChange)

        self.TrtChange()

    def TrtChange(self):
        TrtName = self.CmbTrts.currentText()
        
        self.CmbChar.clear()
        for CName in sorted(self.Data[TrtName]):
            self.CmbChar.addItem(CName)

        PyFETPlot.PlotACNoise(self.Data[TrtName] ,self.AxsNo)
        self.FigNo.canvas.draw()
        self.UpdateAll()
        
    def CharChange(self):
        self.UpdateAll()
        
    def VdChange(self):
        self.UpdateAll()

    def VgChange(self):
        self.UpdateAll()

    def UpdateAll(self):
        TrtName = self.CmbTrts.currentText()
        CName = self.CmbChar.currentText()
        TmpDat = {}
        
        Name = '{}-{}'.format(self.Data[TrtName][CName]['Name'],CName)        
        TmpDat[Name] = self.Data[TrtName][CName]
                
        self.SpinVd.setMaximum(len(TmpDat[Name]['Vds'])-1)
        self.SpinVg.setMaximum(len(TmpDat[Name]['Vgs'])-1)
        
        iVg = self.SpinVg.value()
        iVd = self.SpinVd.value()
        
        Inf = '{} Vds = {} Vgs = {}'.format(Name,
                                     TmpDat[Name]['Vds'][iVd],
                                     TmpDat[Name]['Vgs'][iVg])
        
        InfPars = 'a = {} b = {} irms = {}'.format(TmpDat[Name]['NoA'][iVg,iVd],
                                                   TmpDat[Name]['NoB'][iVg,iVd],
                                                   TmpDat[Name]['Irms'][iVg,iVd])
        self.LabInfo.setText(Inf)
        self.LabInfoPars.setText(InfPars)
        
        PyFETPlot.PlotAC(TmpDat,self.Axs,iVgs=iVg,iVds=iVd)
        self.Axs['PSD'].set_title(Inf)
        self.fig.canvas.draw()
        
#     
#        
#        
#if __name__ == "__main__":
#    app = QtWidgets.QApplication(sys.argv)
#    window = MyApp()
#    window.show()
#    sys.exit(app.exec_())
#    
    
    