# -*- coding: utf-8 -*-
"""
Created on Tue Feb 06 16:56:50 2018

@author: Javier
"""
import os
import sys

from qtpy.QtWidgets import (QHeaderView, QCheckBox, QSpinBox, QLineEdit,
                            QDoubleSpinBox, QTextEdit, QComboBox,
                            QTableWidget, QAction, QMessageBox, QFileDialog,
                            QInputDialog)

from qtpy import QtWidgets, uic

import matplotlib.pyplot as plt
import PyGFET.StimCore as PyCharact
from PyGFET.RecordPlot import PltSlot, PlotRecord

import pickle
import quantities as pq
import inspect
import matplotlib.cm as cmx
import matplotlib.colors as mpcolors

###############################################################################
######
###############################################################################


class ContinuousAcquisitionPlots():

    def __init__(self, Rec):
        slots = []
        cmap = cmx.ScalarMappable(mpcolors.Normalize(
                                  vmin=0, vmax=len(Rec.SigNames.keys())),
                                  cmx.jet)

        for ind, sign in enumerate(sorted(Rec.SigNames.keys())):
            sl = PltSlot()
            sl.rec = Rec
            sl.Position = 0
            sl.Color = cmap.to_rgba(ind)
            sl.DispName = sign
            sl.SigName = sign
            sl.OutType = 'V'
            slots.append(sl)

        #  Init Plot figures
        self.PltRecs = PlotRecord()
        self.PltRecs.CreateFig(slots, ShowLegend=True)
        plt.show()

    def PlotUpdate(self, Time):
        self.PltRecs.ClearAxes()
        self.PltRecs.PlotChannels(Time, Resamp=True)
        self.PltRecs
        self.PltRecs.Fig.canvas.draw()

    def __del__(self):
        plt.close('all')

###############################################################################
####
###############################################################################


class CharactAPP(QtWidgets.QMainWindow):
    OutFigFormats = ('svg', 'png')

    PlotCont = None
#    PlotSweep = None
    FileName = None

    Charac = None  # intance of charact class

    def InitMenu(self):
        mainMenu = self.menubar
        fileMenu = mainMenu.addMenu('File')

        SaveFigAction = QAction('Save Figures', self)
        SaveFigAction.setShortcut('Ctrl+s')
        SaveFigAction.setStatusTip('Save all open figures')
        SaveFigAction.triggered.connect(self.SaveFigures)
        fileMenu.addAction(SaveFigAction)

        CloseFigsAction = QAction('Close Figures', self)
        CloseFigsAction.setStatusTip('Close all open figures')
        CloseFigsAction.triggered.connect(self.CloseFigures)
        fileMenu.addAction(CloseFigsAction)

        LoadConfAction = QAction('Load Configuration', self)
        LoadConfAction.setStatusTip('Load Config')
        LoadConfAction.triggered.connect(self.LoadConf)
        fileMenu.addAction(LoadConfAction)

        SaveConfAction = QAction('Save Configuration', self)
        SaveConfAction.setStatusTip('Save Config')
        SaveConfAction.triggered.connect(self.SaveConf)
        fileMenu.addAction(SaveConfAction)

    def __init__(self, parent=None):

        QtWidgets.QMainWindow.__init__(self)
        uipath = os.path.join(os.path.dirname(__file__), 'GuiStim.ui')
        uic.loadUi(uipath, self)
        self.setWindowTitle('Stimulation PyFET')

        self.InitMenu()

        # Buttons
        self.ButInitChannels.clicked.connect(self.ButInitChannelsClick)
        self.ButUnselAll.clicked.connect(self.ButUnselAllClick)
        self.ButCont.clicked.connect(self.ButContClick)
        self.ButSaveCont.clicked.connect(self.SaveContData)

        # Slider
        self.SLTstart.valueChanged.connect(self.SLTStartChanged)
        self.SLTstop.valueChanged.connect(self.SLTStopChanged)

        # Spin Box
        self.SpnSVinTP.valueChanged.connect(self.VinTimePlotChanged)

        self.ContEnableObjects = [self.SpnTestFreqMin,
                                  self.SpnTestFreqMax,
                                  self.SpnTestNFreqs,
                                  self.SpnTestAmp,
                                  self.SpnRefresh,
                                  self.SpnFsTime]

# Init Channels
###############################################################################
    def ButUnselAllClick(self):
        for ck in self.GrChannels.findChildren(QtWidgets.QCheckBox):
            ck.setChecked(False)

    def ButInitChannelsClick(self):
        # Event InitChannels button
        Channels = self.GetSelectedChannels(self.GrChannels)
        GateChannels = self.GetSelectedChannels(self.GrChannelGate)
        self.GateCh = GateChannels
        if len(GateChannels) > 0:
            if len(GateChannels) > 1:
                QMessageBox.question(self, 'Message',
                                     "Warning: Select Only ONE Gate!",
                                     QMessageBox.Ok)
                return
            GateChannel = GateChannels[0]
        else:
            GateChannel = None

        if GateChannel in Channels:
            QMessageBox.question(self, 'Message', "Warning: Gate Duplicated!",
                                 QMessageBox.Ok)
            return

        if self.Charac is not None:
            self.Charac.__del__()
        self.Charac = PyCharact.Charact(Channels=Channels, GateChannel=None)

        # Define events callbacks
        self.Charac.EventContinuousDone = self.CharContDataCallback

        # Define Gains
#        self.Charac.IVGainDC = float(self.QGainDC.text())
        self.Charac.Rds = float(self.QRds.text())

    def GetSelectedChannels(self, ChGroup):
        Chs = []
        for ck in ChGroup.findChildren(QtWidgets.QCheckBox):
            if ck.isChecked():
                Chs.append(str(ck.text()))
        return Chs  # Dictat amb els canals ['Ch08', 'Ch16', ...

    def GetConfig(self, ConfGroup):
        Config = []
        for n in ConfGroup.findChildren(QtWidgets.QCheckBox):
            if n.isChecked():
                Config.append(str(n.text()))
        return Config[0]

    def SetEnableObjects(self, val, Objects):
        for obj in Objects:
            obj.setEnabled(val)

# Continuous Acquisition
###############################################################################
    def ButContClick(self):  # Evento button TimeCont
        if self.Charac is None:
            print 'Init Channels first'
            return

        if self.Charac.CharactRunning:
            self.Charac.StopCharac()
            self.ButCont.setText('Start')
            self.SetEnableObjects(val=True, Objects=self.ContEnableObjects)
            self.SaveContData()
            self.Charac.ContRecord = None

        else:
            self.SetEnableObjects(val=False,
                                  Objects=self.ContEnableObjects)
            self.ButCont.setText('Stop')

            if self.PlotCont:
                del self.PlotCont

#            self.SetTestSignalConfig()
            print 'InitContMeas'
            self.Charac.InitContMeas(Vin=self.SpnSVinTP.value(),
                                     Fs=self.SpnFsTime.value(),
                                     Refresh=self.SpnRefresh.value(),
                                     RecDC=True,
                                     GenTestSig=self.chckTestSig.isChecked())
            self.PlotCont = ContinuousAcquisitionPlots(self.Charac.ContRecord)

            if self.Charac.CharactRunning:
                self.ButCont.setText('Stop')
            else:
                print 'ERROR'

#    def SetTestSignalConfig(self):
#        print 'Gui SetTestSignalConfig'
#        self.Charac.SetContSig(FreqMin=self.SpnTestFreqMin.value(),
#                               FreqMax=self.SpnTestFreqMax.value(),
#                               nFreqs=self.SpnTestNFreqs.value(),
#                               Arms=self.SpnTestAmp.value())

    def VinTimePlotChanged(self):
        if self.Charac.CharactRunning:
            self.Charac.SetBias(Vsig=self.SpnSVinTP.value())

    def SLTStartChanged(self):
        if self.SLTstop.value() <= self.SLTstart.value():
            self.SLTstop.setValue(self.SLTstart.value()+self.SpnWindow.value())
        self.SLTStopChanged()

    def SLTStopChanged(self):
        if self.SLTstop.value() <= self.SLTstart.value():
            if self.SLTstop.value() == 0:
                self.SLTstop.setValue(1)
                self.SLTstart.setValue(self.SLTstop.value()-1)
            self.SLTstart.setValue(self.SLTstop.value()-1)
        time = (self.SLTstart.value()*pq.s, self.SLTstop.value()*pq.s)

        if self.ChckPauseCont.isChecked():
            Name = self.Charac.ChNamesList[0]
            tstop = self.Charac.ContRecord.Signal(ChName=Name).t_stop
            self.SLTstart.setMaximum(tstop)
            self.SLTstop.setMaximum(tstop)
            self.LblTstartMax.setText(str(tstop))
            self.LblStopMax.setText(str(tstop))

            self.PlotCont.PltRecs.ClearAxes()
            time = (self.SLTstart.value()*pq.s, self.SLTstop.value()*pq.s)
            self.PlotCont.PlotUpdate(Time=time)

# Events Done
###############################################################################

    def CharContDataCallback(self, tstop):
        print 'Gui Continuous Data Done Callback'
        if not self.ChckPauseCont.isChecked():
            time = (tstop - self.SpnWindow.value()*pq.s, tstop)
            print tstop, time
            if self.PlotCont:
                self.PlotCont.PlotUpdate(Time=time)


# Save Data Events
###############################################################################
    def SaveContData(self):
        name, _ = QFileDialog.getSaveFileName(self, 'Save File')
        if not name:
            return
        else:
            self.Charac.ContRecord.SaveRecord(name + '.h5')

# Configuration & Figures Menu
###############################################################################
    def SaveConf(self):
        fileName, _ = QFileDialog.getSaveFileName(self, "Export Data", "",
                                                  "Pickle Files (*.pkl);; All Files (*)")
        if not fileName:
            return
        self.guisave(fileName)

    def LoadConf(self):
        LoadFileName = QFileDialog.getOpenFileName(self)
        if LoadFileName[0]:
            self.guirestore(LoadFileName[0])

    def guisave(self, FileName):

        Configuration = {}
        for name, obj in inspect.getmembers(self):
            if isinstance(obj, QCheckBox):
                Configuration[obj.objectName()] = obj.checkState()
            elif isinstance(obj, QSpinBox) or isinstance(obj, QDoubleSpinBox):
                Configuration[obj.objectName()] = obj.value()
            elif isinstance(obj, QLineEdit):
                Configuration[obj.objectName()] = obj.text()
            elif isinstance(obj, QComboBox):
                Configuration[obj.objectName()] = obj.currentIndex()
        pickle.dump(Configuration, open(FileName, 'wb'))

    def guirestore(self, LoadFileName):

        Configuration = pickle.load(open(LoadFileName))
        for nom, obj in inspect.getmembers(self):

            if isinstance(obj, QCheckBox):
                if obj.objectName() in Configuration:
                    obj.setCheckState(Configuration[obj.objectName()])

            elif isinstance(obj, QSpinBox) or isinstance(obj, QDoubleSpinBox):
                if obj.objectName() in Configuration:
                    obj.setValue(Configuration[obj.objectName()])

            elif isinstance(obj, QLineEdit):
                if obj.objectName() in Configuration:
                    obj.setText(Configuration[obj.objectName()])

            elif isinstance(obj, QComboBox):
                if obj.objectName() in Configuration:
                    obj.setCurrentIndex(Configuration[obj.objectName()])

    def CloseFigures(self):
        plt.close('all')

    def SaveFigures(self):
        Dir = QFileDialog.getExistingDirectory(self)
        Prefix, okPressed = QInputDialog.getText(self,
                                                 'Prefix',
                                                 'Prefix for files',
                                                 text='Figure')
        if Dir and okPressed:
            for i in plt.get_fignums():
                plt.figure(i)
                for ext in self.OutFigFormats:
                    fileOut = Dir + '/' + Prefix + '{}.' + ext
                    plt.savefig(fileOut.format(i))


def main():
    import argparse
    import pkg_resources

    # Add version option
    __version__ = pkg_resources.require("PyGFET")[0].version
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', action='version',
                        version='%(prog)s {version}'.format(
                            version=__version__))
    parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    w = CharactAPP()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

