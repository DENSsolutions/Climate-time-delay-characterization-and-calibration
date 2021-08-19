"""
Time Delay Curve Characterization Script
Version 1.5
Wed Aug 11 10:20:00 2021

@author: Merijn Pen
"""

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 8})

#import impulsePy as impulse
import ImpulsePySim as impulse # Simulator

from datetime import datetime
from scipy import optimize
import pandas as pd
import numpy as np
import seaborn as sns

updateDelay = 0.01

# Parameters
temperature = 300
pressureSetpoints = [700]
pressureOffsetSetpoints = [700, 600, 500, 400, 300]
iterations = 3 # Number of measurements per pressure offset

gasStateA = [0.25, 'Reactor', 0, 'Exhaust', 5, 'Reactor']
gasStateB = [1, 'Reactor', 0, 'Exhaust', 4.25, 'Reactor']
gasStates = [gasStateA, gasStateB]

prePar = {
    "parameter": 'gas1FlowMeasured',
    "position" : 0,
    'rollingAverageWindow' : 1,
    "changeTreshold" : 0.01,
    "stableTreshold" : 0.008,
    "minStableDuration" : 15,
    "data" : impulse.gas.data
    
    }

inPar = {
    "parameter" : 'powerMeasured',
    "position" : 1,
    'rollingAverageWindow' : 10,
    "changeTreshold" : 0.002,
    "stableTreshold" : 0.001,
    "minStableDuration" : 15,
    "data" : impulse.heat.data
    }

postPar = {
    "parameter" : 'Methane',
    "position" : 2,
    'rollingAverageWindow' : 20,
    "changeTreshold" : 0.2e-8,
    "stableTreshold" : 0.1e-8,
    "minStableDuration" : 15,
    "data" : impulse.gas.msdata   
    }

allParInfo = [prePar, inPar, postPar]

# Flow-Delay curve function
def curveFunc(x, a, b, c):
    return a * np.exp(-b * x) + c

initialPtICurvePars = [0,0.17,-0.2]
initialItPCurvePars = [0,0.07,0.01]

timePar = 'experimentDuration'
visibleHistory = 300 #Seconds visible in the real-time graphs

# Define colors for the graphs
colors = sns.color_palette("hls", 8) # Colors for the real-time graphs
colorGroups = {}
colorMaps = ['flare' , 'crest']
for idx, pressure in enumerate(pressureSetpoints):
    colorMap = colorMaps[idx]
    colorPallette = sns.color_palette(colorMap, len(pressureOffsetSetpoints)*iterations)
    colorGroups[pressure]=colorPallette

# Create time delay data and curve result DataFrames
timeDelayData = pd.DataFrame(columns = ['Cid', 'Pressure', 'PressureOffset', 'Flow', 'setpointChangeTime', 'preChangeTime', 'inChangeTime', 'postChangeTime', 'PtI', 'ItP'])
timeDelayCurves = pd.DataFrame(index = pressureSetpoints , columns = ['PtI', 'ItP'])

# Subscribe
impulse.heat.data.subscribe()
impulse.gas.data.subscribe()
impulse.gas.msdata.subscribe()
impulse.waitForControl()
impulse.sleep(3) # Wait 3 seconds to make sure that there are measurements available

class dataProcessor():
    def __init__(self, parInfo):
        
        self.parInfo = parInfo
        self.state = "start"
        
        self.dataSource = self.parInfo['data']
        self.processedData = self.dataSource.getDataFrame()
        while self.processedData.shape[0]<1:
            print(f"Not enough {parInfo['parameter']} data")
            impulse.sleep(1)
            self.processedData = self.dataSource.getDataFrame()
            
        self.processedData['Ra'] = self.processedData[self.parInfo['parameter']].rolling(window=self.parInfo['parameter'], win_type='triang', center=True).mean()
        self.processedData['absRaDiff']=abs(self.processedData['Ra'].diff())
        self.processedData['diff']=self.processedData[self.parInfo['parameter']].diff()
        self.processedData['diffSum']=self.processedData['diff'].rolling(window=self.parInfo['rollingAverageWindow'], center=True).sum()
        self.processedData['absDiffSum']= abs(self.processedData['diffSum'])
        
        self.lastSNFilter = 0
        self.lastSNRA = 0
        self.lastSN = self.processedData.iloc[-1]['sequenceNumber']
    
    def processNew(self):
        newSN = self.dataSource.getLastData()['sequenceNumber']
        newRows = newSN-self.lastSN

        if newRows > 0:
            newRowData = self.dataSource.getDataFrame(-(newRows+20)) # Grab 10 extra rows to make sure that there is no missing rows due to new measurements since newSN was checked
            newRowData['Ra'] = newRowData[self.parInfo['parameter']].rolling(window=self.parInfo['parameter'], win_type='triang', center=True).mean()
            newRowData['absRaDiff']=abs(newRowData['Ra'].diff())
            newRowData['diff']=newRowData[self.parInfo['parameter']].diff()
            newRowData['diffSum']=newRowData['diff'].rolling(window=self.parInfo['rollingAverageWindow'], center=True).sum()
            newRowData['absDiffSum']= abs(newRowData['diffSum'])
            self.processedData = self.processedData.combine_first(newRowData)
            self.lastSN = newSN        

preDataProcessor = dataProcessor(prePar)
inDataProcessor = dataProcessor(inPar)
postDataProcessor = dataProcessor(postPar) # Filter values tested

class createPlotWindow():
    def __init__(self, preDataProcessor, inDataProcessor, postDataProcessor):
        self.preTEMlast = 0
        self.inTEMlast = 0
        self.postTEMlast = 0

        tkw = dict(size=4, width=1.5)

        self.fig = plt.figure(figsize=(13, 10), dpi=80)
        self.fig.canvas.set_window_title('Time Delay Curve Characterization Script')
        plt.tight_layout(pad=0)
        self.fig.suptitle('', fontsize=20)
        
        # Graph window layout
        gsCols = self.fig.add_gridspec(1, 3)
        gs1 = gsCols[0].subgridspec(2, 1)
        gs2 = gsCols[1].subgridspec(3, 1)
        gs3 = gsCols[2].subgridspec(2, 1)
        gsCf = gs1[1].subgridspec(3, 1, hspace=0)
        
        # Realtime measurements plot (top left)
        self.G1P1 = self.fig.add_subplot(gs1[0])
        self.G1P1.margins(y=0)
        self.G1P1.set_title('Measured parameter data')
        self.G1P1.yaxis.label.set_color(colors[0])
        self.G1P1.tick_params(axis='y', colors=colors[0], **tkw)
        self.G1P2 = self.G1P1.twinx()
        self.G1P2.yaxis.label.set_color(colors[1])
        self.G1P2.tick_params(axis='y', colors=colors[1], **tkw)
        self.G1P3 = self.G1P1.twinx()
        self.G1P3.yaxis.label.set_color(colors[2])
        self.G1P3.tick_params(axis='y', colors=colors[2], **tkw)    
        
        # Change value plots (bottom left)
        self.P1C = self.fig.add_subplot(gsCf[0], sharex=self.G1P1)
        self.P1C.margins(y=0)
        self.P1C.set_title('Absolute parameter change')   
        self.P1C.yaxis.label.set_color(colors[0])
        self.P1C.tick_params(axis='y', colors=colors[0], **tkw)
        plt.setp(self.P1C.get_xticklabels(), visible=False)       
        self.P2C = self.fig.add_subplot(gsCf[1], sharex=self.G1P1)
        self.P2C.yaxis.label.set_color(colors[0])
        self.P2C.tick_params(axis='y', colors=colors[1], **tkw)
        plt.setp(self.P2C.get_xticklabels(), visible=False)     
        self.P3C = self.fig.add_subplot(gsCf[2], sharex=self.G1P1)
        self.P3C.yaxis.label.set_color(colors[0])
        self.P3C.tick_params(axis='y', colors=colors[2], **tkw)
        
        # Change data graphs (middle)
        self.preGraph = self.fig.add_subplot(gs2[0])
        self.preGraph.set_title('Pre-TEM changes')
        self.inGraph = self.fig.add_subplot(gs2[1])
        self.inGraph.set_title('In-TEM changes')
        self.postGraph = self.fig.add_subplot(gs2[2])
        self.postGraph.set_title('Post-TEM changes')        
        
        # Time delay curve plots (right)
        self.curv1 = self.fig.add_subplot(gs3[0])
        self.curv1.set_ylabel("Delay (s)")
        self.curv1.set_xlabel("Flow (ml/min)")
        self.curv1.set_title('Pre-to-In curve')
        self.curv2 = self.fig.add_subplot(gs3[1])
        self.curv2.set_ylabel("Delay (s)")
        self.curv2.set_xlabel("Flow (ml/min)")
        self.curv2.set_title('In-to-Post curve')

        # Put the plots in lists for easy access
        self.parameterPlots = [self.G1P1, self.G1P2, self.G1P3]
        self.parameterChangePlots = [self.P1C, self.P2C, self.P3C]
        self.changeGraphs = [self.preGraph, self.inGraph, self.postGraph]

        # Put the dataProcessors in a list for easy access
        self.dataProcessors = [preDataProcessor, inDataProcessor, postDataProcessor]
        self.dataBases = [preDataProcessor.processedData, inDataProcessor.processedData, postDataProcessor.processedData]

        plt.show()


    def setStatus(self, testno, totaltests, status):
        self.fig.suptitle("Test "+str(testno)+"/"+str(totaltests)+' status: '+status, fontsize=20)


    def updateChangePlots(self):      
        extraSpaceRear = 25
        if timeDelayData.shape[0]>0:
            
            for graph in self.changeGraphs:
                graph.cla()
    
            self.preGraph.set_title('Pre-TEM changes')
            self.inGraph.set_title('In-TEM changes')
            self.postGraph.set_title('Post-TEM changes')

            for index, row in timeDelayData.iterrows():
                if row.notna()['Cid']:
                    if index==timeDelayData.shape[0]-1: 
                        opacity = 1
                        linewidth = 2
                    else: 
                        opacity = 0.5
                        linewidth = 1

                    Cid = row['Cid']
                    colorGroup = colorGroups[row['Pressure']]
                    colorIndex = (pressureOffsetSetpoints.index(row['PressureOffset'])*iterations)+row['It']
                    colorIndex = int(colorIndex)
                    color = colorGroup[colorIndex]
                    if row.notna()['preChangeTime']:
                        changeTime = row['preChangeTime']
                        data = impulse.gas.data.getDataFrame(Cid)
                        data = data[data[timePar]<=changeTime+extraSpaceRear]
                        if data.shape[0]>5:
                            xdata = data[timePar]
                            ydata = data[self.dataProcessors[0].parInfo['parameter']]
                            xoffset = [x - row['preChangeTime'] for x in xdata]
                            line,  = self.preGraph.plot(xoffset, ydata, label='Change ' + Cid, color=color, alpha=opacity, linewidth=linewidth, linestyle='-')
                            handles, labels = self.preGraph.get_legend_handles_labels()
                            self.preGraph.legend(handles=handles[-5:], loc='lower right')
        
                    if row.notna()['inChangeTime']:
                        changeTime = row['inChangeTime']
                        data = impulse.heat.data.getDataFrame(Cid)
                        data = data[data[timePar]<=changeTime+extraSpaceRear]
                        if data.shape[0]>5:
                            xdata = data[timePar]
                            ydata = data[self.dataProcessors[1].parInfo['parameter']]
                            xoffset = [x - row['preChangeTime'] for x in xdata]                                      
                            line,  = self.inGraph.plot(xoffset, ydata, label='Change ' + Cid,color=color, alpha=opacity, linewidth=linewidth, linestyle='-')
                            self.inGraph.vlines(changeTime-row['preChangeTime'], ymin=self.inGraph.get_ylim()[0], ymax=self.inGraph.get_ylim()[1], colors=line.get_color(), alpha=opacity, linewidth=linewidth, linestyle='dashed')
                            handles, labels = self.inGraph.get_legend_handles_labels()
                            self.inGraph.legend(handles=handles[-5:], loc='lower left')
        
                    if row.notna()['postChangeTime']:
                        changeTime = row['postChangeTime']
                        data = impulse.gas.msdata.getDataFrame(Cid)
                        data = data[data[timePar]<=changeTime+extraSpaceRear]
                        if data.shape[0]>5:
                            xdata = data[timePar]
                            ydata = data[self.dataProcessors[2].parInfo['parameter']]
                            xoffset = [x - row['preChangeTime'] for x in xdata]
                            line,  = self.postGraph.plot(xoffset, ydata, label='Change ' + Cid,color=color, alpha=opacity, linewidth=linewidth, linestyle='-')
                            self.postGraph.vlines(changeTime-row['preChangeTime'], ymin=self.postGraph.get_ylim()[0], ymax=self.postGraph.get_ylim()[1], colors=line.get_color(), alpha=opacity, linewidth=linewidth, linestyle='dashed')
                            
                            handles, labels = self.postGraph.get_legend_handles_labels()
                            self.postGraph.legend(handles=handles[-5:], loc='lower left')
            
            self.preGraph.vlines(0, ymin=self.preGraph.get_ylim()[0], ymax=self.preGraph.get_ylim()[1], colors="grey", alpha=1, linewidth=1, linestyle='dashed')
            self.inGraph.vlines(0, ymin=self.inGraph.get_ylim()[0], ymax=self.inGraph.get_ylim()[1], colors="grey", alpha=1, linewidth=1, linestyle='dashed')
            self.postGraph.vlines(0, ymin=self.postGraph.get_ylim()[0], ymax=self.postGraph.get_ylim()[1], colors="grey", alpha=1, linewidth=1, linestyle='dashed')


    def updateDelayCurves(self):
        while self.curv1.lines:
            self.curv1.lines.pop(0)
        while self.curv2.lines:
            self.curv2.lines.pop(0)
             
        for index, row in timeDelayData.iterrows():
            colorGroup = colorGroups[row['Pressure']]
            colorIndex = (pressureOffsetSetpoints.index(row['PressureOffset'])*iterations)+row['It']
            colorIndex = int(colorIndex)
            color = colorGroup[colorIndex]
            xpoint = row['Flow']
            y1point = row['PtI']
            y2point = row['ItP']
            self.curv1.plot(xpoint,y1point, 'o', color=color)
            self.curv2.plot(xpoint,y2point, 'o', color=color)
        

        for pressureVal, row in timeDelayCurves.iterrows():
            pressure = pressureVal
            color = colorGroups[pressureVal][0]
            curveDataOrdered = timeDelayData.sort_values(by="Flow", ascending=True)
            if row.notna()['PtI']:
                curveX = np.linspace(curveDataOrdered['Flow'].min(),curveDataOrdered['Flow'].max(),10)
                ptiCurveParams = row['PtI']
                curve1Y = curveFunc(curveX, *ptiCurveParams)
                self.curv1.plot(curveX, curve1Y, 'r-', color=color, label='P:' + str(pressure) + ' fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(ptiCurveParams))
                self.curv1.set_ylim(curve1Y.min(), curve1Y.max())
                self.curv1.legend(loc='upper right')
            if row.notna()['ItP']:
                curveX = np.linspace(curveDataOrdered['Flow'].min(),curveDataOrdered['Flow'].max(),10)
                itpCurveParams = row['ItP']
                curve2Y = curveFunc(curveX, *itpCurveParams)
                self.curv2.plot(curveX, curve2Y, 'r-', color=color, label='P:' + str(pressure) + ' fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(itpCurveParams))
                self.curv2.set_ylim(curve2Y.min(), curve2Y.max())
                self.curv2.legend(loc='upper right')

        plt.pause(updateDelay)


    def updateRtPlots(self):
        plot1Legend = []
        for idx, par in enumerate(allParInfo):
            color = par['position']
            parameter = par['parameter']
            if color == 0:
                plotData = preDataProcessor.processedData[preDataProcessor.processedData[timePar]>preDataProcessor.processedData.iloc[-1][timePar]-visibleHistory]
            elif color == 1:
                plotData = inDataProcessor.processedData[inDataProcessor.processedData[timePar]>inDataProcessor.processedData.iloc[-1][timePar]-visibleHistory]
            elif color == 2:
                plotData = postDataProcessor.processedData[postDataProcessor.processedData[timePar]>postDataProcessor.processedData.iloc[-1][timePar]-visibleHistory]
                       
            while self.parameterPlots[idx].lines:
                self.parameterPlots[idx].lines.pop(0)
    
            if 'filtered' in plotData.columns:
                y = plotData['filtered']
                x = plotData[timePar]
                line, = self.parameterPlots[idx].plot(x, y, color=colors[color], label='Filtered '+parameter, alpha=1, linewidth=2, linestyle='solid')
                ymin = y.min()-0.1*(y.max()-y.min())
                ymax = y.max()+0.1*(y.max()-y.min())
                if ymin != ymax and len(y)>0 and not np.isnan(ymin) and not np.isnan(ymax): self.parameterPlots[idx].set_ylim(ymin, ymax)
                y = plotData[parameter]
                x = plotData[timePar]
                self.parameterPlots[idx].plot(x, y, color=colors[color], label=parameter, alpha=0.2, linewidth=1, linestyle='solid')
                plot1Legend = plot1Legend + [line]

            elif parameter in plotData.columns:
                y = plotData[parameter]
                x = plotData[timePar]
                line, = self.parameterPlots[idx].plot(x, y, color=colors[color], label=parameter,  alpha=1, linewidth=2, linestyle='solid')
                ymin = y.min()-0.1*(y.max()-y.min())
                ymax = y.max()+0.1*(y.max()-y.min())
                if ymin != ymax: self.parameterPlots[idx].set_ylim(ymin, ymax)
                self.parameterPlots[idx].set_xlim(plotData[timePar].min(),plotData[timePar].max())
                plot1Legend = plot1Legend + [line]
            
            #Empty the changeVec plot
            while self.parameterChangePlots[idx].lines:
                self.parameterChangePlots[idx].lines.pop(0)
            
            if 'absDiffSum' in plotData.columns:
                y = plotData['absDiffSum']
                x = plotData[timePar]
                self.parameterChangePlots[idx].plot(x, y, color=colors[color], label=parameter, alpha=1, linewidth=2, linestyle='-')
                self.parameterChangePlots[idx].axhline(par['changeTreshold'], color=colors[color], label='Change treshold: '+ str(par['changeTreshold']), alpha=0.5, linestyle='dashed')
                self.parameterChangePlots[idx].axhline(par['stableTreshold'], color=colors[color], label='Stable treshold: '+ str(par['stableTreshold']), alpha=0.5, linestyle='dotted')
                self.parameterChangePlots[idx].set_xlim(plotData[timePar].min(),plotData[timePar].max())
                self.parameterChangePlots[idx].legend(loc='upper left')

                if plotData['absDiffSum'].max()>par['changeTreshold']:
                    ymax = plotData['absDiffSum'].max()
                else:
                    ymax = par['changeTreshold']
                self.parameterChangePlots[idx].set_ylim(0, ymax)
                
        self.parameterPlots[2].legend(plot1Legend, [allParInfo[0]['parameter'], allParInfo[1]['parameter'], allParInfo[2]['parameter']])
    
    
    def updatePlots(self):    
        self.updateRtPlots()
        self.updateChangePlots()        
        self.fig.canvas.draw()
        plt.pause(updateDelay)
    
plotPanel = createPlotWindow(preDataProcessor, inDataProcessor, postDataProcessor)
   

class controller():
    def __init__(self, preDataCollector, inDataCollector, postDataCollector):
        self.dataCollectors = [preDataCollector,inDataCollector,postDataCollector]
        self.state = 0
        self.currentInletPressure = 0
        self.currentOutletPressure = 0
        self.testActive = True
        self.lastCheckedTime = 0
        self.detectChangeTimeoutCounter = 0
        self.detectChangeTimeoutVal = 2000
        self.noiseRangeMeasurementStart = 0
        
        # New way of controlling the tests
        self.sequence = pd.DataFrame(columns = ['P', 'PO', 'It', 'State'])
        state = 0
        for p in pressureSetpoints:
            nP = 1
            for po in pressureOffsetSetpoints:
                nPO = 1
                for i in range(iterations):
                    self.sequence = self.sequence.append({'P' : p, 'nP' : nP, 'PO' : po, 'nPO' : nPO, 'It' : i, 'State' : state}, ignore_index=True)
                    nP =0
                    nPO=0
                    state = abs(state-1) #Toggle between 0 and 1
        self.sequenceStep = 0

        self.checkChangePos = 0
        self.checkChangePositions = {
            0 : "preChangeTime",
            1 : "inChangeTime",
            2 : "postChangeTime"
            }
    
    def fitCurves(self):
        print("Fitting curves")
        plotPanel.setStatus(self.sequenceStep+1,self.sequence.shape[0],'Fitting curves')
        pressures = timeDelayData['Pressure'].unique()
        for pressure in pressures:
            pressureDelayData = timeDelayData[timeDelayData['Pressure']==pressure]
            if len(pressureDelayData['PressureOffset'].unique())>2:
                pressureDelayDataOrdered  = pressureDelayData.sort_values(by="Flow", ascending=True)          
                flowData = pressureDelayDataOrdered['Flow'].tolist()
                ptiData = pressureDelayDataOrdered["PtI"].tolist()
                itpData = pressureDelayDataOrdered["ItP"].tolist()
                try:
                    self.ptiCurveParams, ptiCurveCovar = optimize.curve_fit(curveFunc, flowData, ptiData, p0=initialPtICurvePars, maxfev=1000)
                    timeDelayCurves.loc[pressure,'PtI']=self.ptiCurveParams
                except:
                    print("Could not fit curve to PtI data (yet)")
                try:
                    self.itpCurveParams, itpCurveCovar = optimize.curve_fit(curveFunc, flowData, itpData, p0=initialItPCurvePars, maxfev=1000)
                    timeDelayCurves.loc[pressure,'ItP']=self.itpCurveParams
                except:
                    print("Could not fit curve to ItP data (yet)")
            plotPanel.updateDelayCurves()
    
    
    def setPressureConditions(self):
        newState = self.sequence.iloc[self.sequenceStep]
        pressure = newState['P']
        pressureOffset = newState['PO']
        self.currentInletPressure = pressure+(0.5*pressureOffset)
        self.currentOutletPressure = pressure-(0.5*pressureOffset)
        print(f"Setting pressures {self.currentInletPressure} - {self.currentOutletPressure}")
        plotPanel.setStatus(self.sequenceStep+1,self.sequence.shape[0],f"Setting pressures {self.currentInletPressure} - {self.currentOutletPressure}")
        gasMixState = gasStates[abs(newState['State']-1)] # This is ugly, but we need to set the opposite gas state first
        impulse.gas.setIOP(self.currentInletPressure,self.currentOutletPressure,gasMixState[0],gasMixState[1],gasMixState[2],gasMixState[3],gasMixState[4],gasMixState[5])
        return 1    

        
    def waitForPressures(self):
        pressureMargin = 5 #mbar
        print("Waiting to reach pressures")
        plotPanel.setStatus(self.sequenceStep+1,self.sequence.shape[0],'Waiting to reach pressures')
        if abs(impulse.gas.data.getLastData()['inletPressureMeasured']-impulse.gas.data.getLastData()['inletPressureSetpoint'])<pressureMargin and abs(impulse.gas.data.getLastData()['outletPressureMeasured']-impulse.gas.data.getLastData()['outletPressureSetpoint'])<pressureMargin:
            return 1
        else:
            return 0


    def waitUntilStable(self):
        print("Waiting for stable conditions")
        plotPanel.setStatus(self.sequenceStep+1,self.sequence.shape[0],"Waiting for stable conditions")
        
        if any([(self.dataCollectors[0].processedData[self.dataCollectors[0].processedData['absDiffSum'].notna()].shape[0]<self.dataCollectors[0].parInfo['minStableDuration']),
               (self.dataCollectors[1].processedData[self.dataCollectors[1].processedData['absDiffSum'].notna()].shape[0]<self.dataCollectors[1].parInfo['minStableDuration']),
               (self.dataCollectors[2].processedData[self.dataCollectors[2].processedData['absDiffSum'].notna()].shape[0]<self.dataCollectors[2].parInfo['minStableDuration'])
               ]):
            print("Not enough changevector data yet...")
        
        elif 'absDiffSum' in self.dataCollectors[0].processedData.columns and 'absDiffSum' in self.dataCollectors[1].processedData.columns and 'absDiffSum' in self.dataCollectors[2].processedData.columns:
            if self.dataCollectors[0].processedData.tail(self.dataCollectors[0].parInfo['minStableDuration'])['absDiffSum'].max() < self.dataCollectors[0].parInfo['stableTreshold']:
                if self.dataCollectors[1].processedData.tail(self.dataCollectors[1].parInfo['minStableDuration'])['absDiffSum'].max() < self.dataCollectors[1].parInfo['stableTreshold']:
                    if self.dataCollectors[2].processedData[self.dataCollectors[2].processedData['absDiffSum'].notna()].tail(self.dataCollectors[2].parInfo['minStableDuration'])['absDiffSum'].max() < self.dataCollectors[2].parInfo['stableTreshold']:
                        self.dataCollectors[0].state = "stable"
                        self.dataCollectors[1].state = "stable"
                        self.dataCollectors[2].state = "stable"
                        print("All are stable!")
                        plotPanel.setStatus(self.sequenceStep+1,self.sequence.shape[0],"All 3 parameters are stable!")
                        return 1
                    else:
                        print("Post-TEM not stable")
                        return 0
                else:
                    print("In-TEM not stable")
                    return 0
            else:
                print("Pre-TEM not stable")
                return 0
        else:
            print("Did not find the columns")
        return 0
    
    
    def measureNoiseRange(self):
        noiseRangeMeasurementLength = 10 #seconds
        if self.noiseRangeMeasurementStart == 0:
            plotPanel.setStatus(self.sequenceStep+1,self.sequence.shape[0],"Measuring noise range")
            self.noiseRangeMeasurementStart = self.dataCollectors[2].processedData.iloc[-1][timePar]
            return 0
        else:
            noiseRanges = []
            notEnoughData = False
            for i in range(3):            
                measurementData = self.dataCollectors[i].processedData[self.dataCollectors[i].processedData[timePar]>=self.noiseRangeMeasurementStart]
                checkPar = self.dataCollectors[i].parInfo['parameter']
                if 'filtered' in self.dataCollectors[i].processedData.columns: checkPar = 'filtered'
                measurementDataNotna = measurementData[measurementData[checkPar].notna()]
                if measurementDataNotna.shape[0]>1:
                    if measurementDataNotna.iloc[-1][timePar]-measurementDataNotna.iloc[0][timePar]>noiseRangeMeasurementLength:
                        minVal = measurementDataNotna[checkPar].min()
                        maxVal = measurementDataNotna[checkPar].max()
                        offset = (maxVal-minVal)/5 # Some margin in case the signal has drifted outside of the original boundaries. If the start of the change is detected too far ahead of the change, then increase this value.
                        noiseRange = [minVal-offset, maxVal+offset]
                        noiseRanges.extend([noiseRange])                     
                    else: 
                        notEnoughData = True
                        break
                else:
                    notEnoughData = True
                    break
            if notEnoughData == False:
                self.noiseRanges = noiseRanges
                self.noiseRangeMeasurementStart = 0
                return 1
            else:
                return 0

                
    def initiateChange(self):
        print("Initiating gas change")
        plotPanel.setStatus(self.sequenceStep+1,self.sequence.shape[0],"Initiating gas change")
        timeDelayData.at[self.sequenceStep,'Pressure']=self.sequence.iloc[self.sequenceStep]['P']
        timeDelayData.at[self.sequenceStep,'PressureOffset']=self.sequence.iloc[self.sequenceStep]['PO']
        timeDelayData.at[self.sequenceStep,'It']=self.sequence.iloc[self.sequenceStep]['It']
        flagName = 'F'+str(self.sequence.iloc[self.sequenceStep]['P'])+str(self.sequence.iloc[self.sequenceStep]['PO'])+str(self.sequence.iloc[self.sequenceStep]['It'])
        impulse.gas.data.setFlag(flagName)
        impulse.heat.data.setFlag(flagName)
        impulse.gas.msdata.setFlag(flagName)
        initiateTime = impulse.gas.data.getNewData()[timePar]
        self.lastCheckedTime = initiateTime
        currentSetpoint = impulse.gas.data.getLastData()
        if currentSetpoint['gas1FlowSetpoint']!=gasStateA[0]:
            returnMessage = impulse.gas.setIOP(self.currentInletPressure,self.currentOutletPressure,gasStateA[0],gasStateA[1],gasStateA[2],gasStateA[3],gasStateA[4],gasStateA[5])
        else:
            returnMessage = impulse.gas.setIOP(self.currentInletPressure,self.currentOutletPressure,gasStateB[0],gasStateB[1],gasStateB[2],gasStateB[3],gasStateB[4],gasStateB[5])
        if returnMessage != "ok":
            print(f"Gas state not accepted!\n Returnmessage: {returnMessage}")
        timeDelayData.at[self.sequenceStep,'setpointChangeTime']=initiateTime      
        self.checkChangePos = 0
        return 1

    
    def detectChange(self):
        dataCollector = self.dataCollectors[self.checkChangePos]
        startChangeTime = 0
        locations = ["pre-TEM", "in-TEM", "post-TEM"]
        print(f"Detecting change {locations[self.checkChangePos]}")
        plotPanel.setStatus(self.sequenceStep+1,self.sequence.shape[0],f"Detecting change {locations[self.checkChangePos]}")
        indices = dataCollector.processedData[dataCollector.processedData[timePar].gt(self.lastCheckedTime)].index
        foundLast = 0
        
        if len(indices)>1:
            startRow = indices[0]
            endRow = indices[-1]
            for i in np.arange(startRow, endRow, 1):
                if not np.isnan(dataCollector.processedData.loc[i]['absDiffSum']):
                    if dataCollector.processedData.loc[i-dataCollector.parInfo['minChangeDuration']:i]['absDiffSum'].min() > dataCollector.parInfo['changeTreshold'] and dataCollector.state != "changing":
                        print(dataCollector.parInfo['parameter'] + "change detected!")
                        plotPanel.setStatus(self.sequenceStep+1,self.sequence.shape[0],dataCollector.parInfo['parameter'] + "change detected!")
                        dataCollector.state = "changing"
                        
                        # Determine when the signal started changing
                        x=i
                        startChangeTime = dataCollector.processedData.loc[x][timePar]
                    
                        checkPar = dataCollector.parInfo['parameter']
                        if 'filtered' in dataCollector.processedData.columns: checkPar = 'filtered'
                        
                        position = dataCollector.parInfo['position']
                        if dataCollector.processedData.loc[x][checkPar]>self.noiseRanges[position][1]:
                            while dataCollector.processedData.loc[x][checkPar]>self.noiseRanges[position][1]:
                                x-=1
                            # while dataCollector.processedData.loc[x][checkPar]<dataCollector.processedData.loc[x][checkPar]:
                            #     x-=1
                            startChangeTime = dataCollector.processedData.loc[x][timePar]
                            
                        else:
                            while dataCollector.processedData.loc[x][checkPar]<self.noiseRanges[position][0]:
                                x-=1
                            # while dataCollector.processedData.loc[x][checkPar]>dataCollector.processedData.loc[x][checkPar]:
                                #     x-=1
                            startChangeTime = dataCollector.processedData.loc[x][timePar]                       
                        
                        
                        timeDelayData.at[self.sequenceStep,self.checkChangePositions[self.checkChangePos]]=startChangeTime
                        flagName = 'F'+str(self.sequence.iloc[self.sequenceStep]['P'])+str(self.sequence.iloc[self.sequenceStep]['PO'])+str(self.sequence.iloc[self.sequenceStep]['It'])
                        timeDelayData.at[self.sequenceStep,'Cid']=flagName
                        
                        # When the postTem change has been detected, store the average Flow of the test
                        if self.checkChangePos == 2:
                            meanFlow = impulse.gas.data.getDataFrame(flagName)['reactorFlowMeasured'].mean()
                            timeDelayData.at[self.sequenceStep,'Flow']=meanFlow
                            timeDelayData.at[self.sequenceStep,'PtI']=timeDelayData.loc[self.sequenceStep,'inChangeTime']-timeDelayData.loc[self.sequenceStep,'preChangeTime']
                            timeDelayData.at[self.sequenceStep,'ItP']=timeDelayData.loc[self.sequenceStep,'postChangeTime']-timeDelayData.loc[self.sequenceStep,'inChangeTime']
                            self.fitCurves()
                            foundLast = 1
                        else:
                            print(f"Found change {self.checkChangePos} at time {startChangeTime}")
                            self.checkChangePos +=1
                        break
                    self.lastCheckedTime = dataCollector.processedData.loc[i][timePar]
        return foundLast
    
    
    def saveData(self):
        now = datetime.now()
        dt_string = now.strftime("%d-%m-%Y_%H-%M-%S_")
        timeDelayData.to_csv(dt_string + 'timeDelayData.csv')
        timeDelayCurves.to_csv(dt_string + 'timeDelayCurvePars.csv')     
    
    
    def work(self):
        if self.state == 0: # Start next state
            if self.sequence.iloc[self.sequenceStep]['nPO']==1: # New combination of pressures
                self.state += self.setPressureConditions()            
            else:
                self.state = 3 # Skip pressure changes and initiate change (as this is just another iteration)        
        elif self.state == 1: 
            self.state += self.waitForPressures()        
        elif self.state == 2:
            self.state += self.waitUntilStable()  
        elif self.state == 3:
            self.state += self.measureNoiseRange()  
        elif self.state == 4:
            self.state += self.initiateChange()      
        elif self.state == 5: 
            self.state += self.detectChange()
            self.detectChangeTimeoutCounter +=1
            if self.detectChangeTimeoutVal - self.detectChangeTimeoutCounter < 10:
                print(f"Timeout in {self.detectChangeTimeoutVal - self.detectChangeTimeoutCounter}")
                plotPanel.setStatus(self.sequenceStep+1,self.sequence.shape[0],f"Timeout in {self.detectChangeTimeoutVal - self.detectChangeTimeoutCounter}")
                if self.detectChangeTimeoutVal == self.detectChangeTimeoutCounter:
                    print("Change not detected before timeout, restarting measurement")
                    plotPanel.setStatus(self.sequenceStep+1,self.sequence.shape[0],"Change not detected before timeout, restarting measurement")
                    self.state = 0
        elif self.state == 6:
            if self.sequenceStep+1 > self.sequence.shape[0]-1:
                print("Last test finished")
                plotPanel.setStatus(self.sequenceStep+1,self.sequence.shape[0],"Last test finished.")
                self.testActive = False
            elif self.sequenceStep+1 < self.sequence.shape[0]:
                if self.sequence.iloc[self.sequenceStep+1]['nPO']==1: #If the next test will be at new pressures, no need to wait for stable conditions
                    self.state+=1
                else: self.state += self.waitUntilStable()     
        elif self.state == 7: # Update sequence number
            self.sequenceStep += 1
            self.state = 0

controller = controller(preDataProcessor,inDataProcessor,postDataProcessor) # Create the test controller

# Start of test
impulse.heat.set(temperature) # Set the test temperature

while controller.testActive:
    # Collect new measurement data and apply filters / analysis
    preDataProcessor.processNew()
    inDataProcessor.processNew()
    postDataProcessor.processNew()
    
    controller.work() # Perform active step in test process
    
    plotPanel.updatePlots() # Update graphs

impulse.gas.setIOP(0,0,0,'Exhaust',0,'Exhaust',0,'Exhaust') # Stop gas flows
impulse.heat.set(21) # Set the temperature back to RT
controller.saveData() # Save the delay data to a file
impulse.disconnect() # Close the connection with Impulse