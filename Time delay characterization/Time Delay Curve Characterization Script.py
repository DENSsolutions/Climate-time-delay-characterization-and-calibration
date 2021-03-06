"""
Time Delay Curve Characterization Script
Version 1.10
Oct 27 2021

@author: Merijn
"""

# Disable the impulsePy import and enable the ImpulsePySim import to use in simulation mode
import impulsePy as impulse
#import ImpulsePySim as impulse # Simulator

# -------- You can change the test parameters below this line: --------

temperature = 300 # The temperature at which all the measurements are taken

pressureSetpoints = [600, 700, 800] # The Nano-Reactor pressures at which time delay measurements are taken. Adviced to set at least 3 pressures.
pressureOffsetSetpoints = [700, 600, 500, 400] # The inlet-outlet pressure offsets at which time delay measurements are taken for each Pnr. Adviced to set at least 4 pressure offsets.
iterations = 1 # Number of measurements per pressure offset

# The two gas states that the script toggles between are defined below:
gasStateA = [0.25, 'Reactor', 0, 'Exhaust', 5, 'Reactor']
gasStateB = [1, 'Reactor', 0, 'Exhaust', 4.25, 'Reactor']

# The next 3 dicts contain settings for the 3 parameters that are monitored:
# Pre-TEM parameter info:
prePar = {
    "parameter": 'gas1FlowMeasured', # Change this if you want to track a different parameter
    "position" : 0, # Do not change this
    'rollingAverageWindow' : 3, # Rolling average window, increase if signal has a high noise level
    "changeThreshold" : 0.01, # Treshold for the change-value above which the signal change is detected
    "stableThreshold" : 0.008, # Treshold for the change-value under which the signal is considered stable
    "minStableDuration" : 15, # Number of measurements that need to be under the stable treshold
    "data" : impulse.gas.data # Do not change this
    }
# In-TEM parameter info:
inPar = {
    "parameter" : 'powerMeasured', # Change this if you want to track a different parameter
    "position" : 1, # Do not change this
    'rollingAverageWindow' : 10, # Rolling average window, increase if signal has a high noise level
    "changeThreshold" : 0.002, # Treshold for the change-value above which the signal change is detected
    "stableThreshold" : 0.001, # Treshold for the change-value under which the signal is considered stable
    "minStableDuration" : 15, # Number of measurements that need to be under the stable treshold
    "data" : impulse.heat.data # Do not change this
    }
# Post-TEM parameter info:
postPar = {
    "parameter" : 'Methane', # Change this if you want to track a different parameter
    "position" : 2, # Do not change this
    'rollingAverageWindow' : 20, # Rolling average window, increase if signal has a high noise level
    "changeThreshold" : 0.2e-8, # Treshold for the change-value above which the signal change is detected
    "stableThreshold" : 0.1e-8, # Treshold for the change-value under which the signal is considered stable
    "minStableDuration" : 15, # Number of measurements that need to be under the stable treshold
    "data" : impulse.gas.msdata # Do not change  this
    }

# -------- No need to change anything after this line --------


import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12, 'axes.linewidth':2, 'xtick.major.width':2, 'ytick.major.width':2})
from datetime import datetime
from scipy import optimize
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import seaborn as sns
from sympy.solvers import solve
from sympy import Symbol, sqrt


updateDelay = 0.01

gasStates = [gasStateA, gasStateB]
allParInfo = [prePar, inPar, postPar]

rollingAverageCenter = True

# Flow-Delay curve function
def FcurveFunc(x, a, b, c, d): #Old function to fit curves for the UI, these are not used for the calibration
    return 1/(a+b*x+c*x**2+d*x**3)

def PFcurveFunc(data, a, b, c, d, e, f): #New function that is used for the calibration
    P=data[0]
    F=data[1]
    return f + (P*e) * (1/(a+b*F+c*F**2+d*F**3))
    
# Calculate Inlet and Outlet setpoints
def calcInletOutletPressures(Pnr, Poff):
    x = Symbol('x')
    Pin = solve(sqrt(0.5*x**2+0.5*(x-Poff)**2)-Pnr, x)
    Pout = solve(sqrt(0.5*(x+Poff)**2+0.5*(x)**2)-Pnr, x)
    if len(Pin)>1: Pin=Pin[1]
    if len(Pout)>1: Pout=Pout[1]
    Pin = int(round(Pin,0))
    Pout = int(round(Pout,0))
    return([Pin,Pout])

timePar = 'experimentDuration'
visibleHistory = 300 #Seconds visible in the real-time graphs

# Define colors for the graphs
colors = sns.color_palette("hls", 8) # Colors for the real-time graphs
colorGroups = {}
colorStep = 3/len(pressureSetpoints)
rotVal = -0.1
for idx, pressure in enumerate(pressureSetpoints):
    colorPallette = sns.cubehelix_palette(start=colorStep*idx, rot=rotVal, light=0.3, dark=0.8, as_cmap=False, n_colors=len(pressureOffsetSetpoints)*iterations)
    colorGroups[pressure]=colorPallette

# Create time delay data and curve result DataFrames
timeDelayData = pd.DataFrame(columns = ['Cid', 'Pressure', 'PressureOffset', 'Flow', 'setpointChangeTime', 'preChangeTime', 'inChangeTime', 'postChangeTime', 'PtI', 'ItP'])
timeDelayFCurves = pd.DataFrame(index = pressureSetpoints , columns = ['PtI', 'ItP'])
timeDelayFCurves['Type']='2D'
timeDelayPFCurves = pd.DataFrame(columns = ['PtI', 'ItP'])
timeDelayPFCurves['Type']='3D'

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
        self.processedData['Ra'] = self.processedData[self.parInfo['parameter']].rolling(window=self.parInfo['rollingAverageWindow'], win_type='triang', center=rollingAverageCenter).mean()
        self.processedData['absRaDiff']=abs(self.processedData['Ra'].diff())
        self.processedData['diff']=self.processedData[self.parInfo['parameter']].diff()
        self.processedData['diffSum']=self.processedData['diff'].rolling(window=self.parInfo['rollingAverageWindow'], center=rollingAverageCenter).sum()
        self.processedData['absDiffSum']= abs(self.processedData['diffSum'])      
        self.lastSNFilter = 0
        self.lastSNRA = 0
        self.lastSN = self.processedData.iloc[-1]['sequenceNumber']
    
    def processNew(self):
        newSN = self.dataSource.getLastData()['sequenceNumber']
        newRows = newSN-self.lastSN
        if newRows > 0:
            newRowData = self.dataSource.getDataFrame(-(newRows+20)) # Grab extra rows to make sure that there is no missing rows due to new measurements since newSN was checked
            newRowData['Ra'] = newRowData[self.parInfo['parameter']].rolling(window=self.parInfo['rollingAverageWindow'], win_type='triang', center=rollingAverageCenter).mean()
            newRowData['absRaDiff']=abs(newRowData['Ra'].diff())
            newRowData['diff']=newRowData[self.parInfo['parameter']].diff()
            newRowData['diffSum']=newRowData['diff'].rolling(window=self.parInfo['rollingAverageWindow'], center=rollingAverageCenter).sum()
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
        extraSpaceFront = 4
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
                        initTime = row['setpointChangeTime']
                        changeTime = row['preChangeTime']
                        data = self.dataProcessors[0].processedData[self.dataProcessors[0].processedData[timePar]>=initTime-extraSpaceFront]
                        data = data[data[timePar]<=changeTime+extraSpaceRear]
                        if data.shape[0]>5:
                            xdata = data[timePar]
                            ydata = data[self.dataProcessors[0].parInfo['parameter']]
                            xoffset = [x - row['preChangeTime'] for x in xdata]
                            line,  = self.preGraph.plot(xoffset, ydata, label='Change ' + Cid, color=color, alpha=opacity, linewidth=linewidth, linestyle='-')
                            handles, labels = self.preGraph.get_legend_handles_labels()
                            self.preGraph.legend(handles=handles[-5:], loc='lower right')
        
                    if row.notna()['inChangeTime']:
                        initTime = row['setpointChangeTime']
                        changeTime = row['inChangeTime']
                        data = self.dataProcessors[1].processedData[self.dataProcessors[1].processedData[timePar]>=initTime-extraSpaceFront]
                        data = data[data[timePar]<=changeTime+extraSpaceRear]
                        if data.shape[0]>5:
                            xdata = data[timePar]
                            ydata = data[self.dataProcessors[1].parInfo['parameter']]
                            xoffset = [x - row['preChangeTime'] for x in xdata]                                      
                            line,  = self.inGraph.plot(xoffset, ydata, label='Change ' + Cid,color=color, alpha=opacity, linewidth=linewidth, linestyle='-')
                            changeY = data[data[timePar]==changeTime][self.dataProcessors[1].parInfo['parameter']]
                            self.inGraph.plot(changeTime-row['preChangeTime'],changeY.values[0], color=line.get_color(), alpha=opacity, marker="d")
                            #self.inGraph.vlines(changeTime-row['preChangeTime'], ymin=self.inGraph.get_ylim()[0], ymax=self.inGraph.get_ylim()[1], colors=line.get_color(), alpha=opacity, linewidth=linewidth, linestyle='dashed')
                            handles, labels = self.inGraph.get_legend_handles_labels()
                            self.inGraph.legend(handles=handles[-5:], loc='lower left')
        
                    if row.notna()['postChangeTime']:
                        initTime = row['setpointChangeTime']
                        changeTime = row['postChangeTime']
                        data = self.dataProcessors[2].processedData[self.dataProcessors[2].processedData[timePar]>=initTime-extraSpaceFront]
                        data = data[data[timePar]<=changeTime+extraSpaceRear]
                        if data.shape[0]>5:
                            xdata = data[timePar]
                            ydata = data[self.dataProcessors[2].parInfo['parameter']]
                            xoffset = [x - row['preChangeTime'] for x in xdata]
                            line,  = self.postGraph.plot(xoffset, ydata, label='Change ' + Cid,color=color, alpha=opacity, linewidth=linewidth, linestyle='-')
                            changeY = data[data[timePar]==changeTime][self.dataProcessors[2].parInfo['parameter']]
                            self.postGraph.plot(changeTime-row['preChangeTime'],changeY.values[0], color=line.get_color(), alpha=opacity, marker="d")

                            #self.postGraph.vlines(changeTime-row['preChangeTime'], ymin=self.postGraph.get_ylim()[0], ymax=self.postGraph.get_ylim()[1], colors=line.get_color(), alpha=opacity, linewidth=linewidth, linestyle='dashed')
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
            
        for pressureVal, row in timeDelayFCurves.iterrows():
            pressure = pressureVal
            color = colorGroups[pressureVal][0]
            curveDataOrdered = timeDelayData.sort_values(by="Flow", ascending=True)
            if row.notna()['PtI']:
                curveX = np.linspace(curveDataOrdered['Flow'].min(),curveDataOrdered['Flow'].max(),20)
                ptiCurveParams = row['PtI']
                curve1Y = FcurveFunc(curveX, *ptiCurveParams)
                self.curv1.plot(curveX, curve1Y, '-', color=color, label='P:' + str(pressure) + ' fit')
                #self.curv1.set_ylim(curve1Y.min(), curve1Y.max())
                self.curv1.legend(loc='upper right')
            if row.notna()['ItP']:
                curveX = np.linspace(curveDataOrdered['Flow'].min(),curveDataOrdered['Flow'].max(),20)
                itpCurveParams = row['ItP']
                curve2Y = FcurveFunc(curveX, *itpCurveParams)
                self.curv2.plot(curveX, curve2Y, '-', color=color, label='P:' + str(pressure) + ' fit')
                #self.curv2.set_ylim(curve2Y.min(), curve2Y.max())
                self.curv2.legend(loc='upper right')

        self.curv1.autoscale()
        self.curv2.autoscale()

        margin = 0.1
        
        ptiHi = timeDelayData['PtI'].max()
        ptiLo = timeDelayData['PtI'].min()
        ptiH = ptiHi-ptiLo
        ptiTop = ptiHi+ptiH*margin
        ptiBottom = ptiLo-ptiH*margin
        self.curv1.set_ylim(ptiBottom,ptiTop)

        itpHi = timeDelayData['ItP'].max()
        itpLo = timeDelayData['ItP'].min()
        itpH = itpHi-itpLo
        itpTop = itpHi+itpH*margin
        itpBottom = itpLo-itpH*margin
        self.curv2.set_ylim(itpBottom,itpTop)

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
    
            style = 'solid'
            if 'Ra' in plotData.columns:
                y = plotData['Ra']
                x = plotData[timePar]
                line, = self.parameterPlots[idx].plot(x, y, color=colors[color], label='RA '+parameter, alpha=1, linewidth=2, linestyle=style)
                ymin = y.min()-0.1*(y.max()-y.min())
                ymax = y.max()+0.1*(y.max()-y.min())
                if ymin != ymax and len(y)>0 and not np.isnan(ymin) and not np.isnan(ymax): self.parameterPlots[idx].set_ylim(ymin, ymax)
                y = plotData[parameter]
                x = plotData[timePar]
                plot1Legend = plot1Legend + [line]
            
            style = 'dotted'
            if parameter in plotData.columns:
                y = plotData[parameter]
                x = plotData[timePar]
                line, = self.parameterPlots[idx].plot(x, y, color=colors[color], label=parameter,  alpha=1, linewidth=2, linestyle=style)
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
                self.parameterChangePlots[idx].axhline(par['changeThreshold'], color='black', label='Change threshold: '+ str(par['changeThreshold']), alpha=0.5, linestyle='dashed')
                self.parameterChangePlots[idx].axhline(par['stableThreshold'], color='black', label='Stable threshold: '+ str(par['stableThreshold']), alpha=0.5, linestyle='dotted')
                self.parameterChangePlots[idx].set_xlim(plotData[timePar].min(),plotData[timePar].max())
                self.parameterChangePlots[idx].legend(loc='upper left')
                self.parameterChangePlots[idx].set_yscale('log')
                
        self.parameterPlots[2].legend(plot1Legend, [f"RA {allParInfo[0]['parameter']}", allParInfo[0]['parameter'], f"RA {allParInfo[1]['parameter']}", allParInfo[1]['parameter'], f"RA {allParInfo[2]['parameter']}", allParInfo[2]['parameter']])
    
    
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
        self.initiateTime=0
        self.lastDetectionTime=0
        
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
                    self.ptiCurveParams, ptiCurveCovar = optimize.curve_fit(FcurveFunc, flowData, ptiData, maxfev=1000)
                    timeDelayFCurves.loc[pressure,'PtI']=self.ptiCurveParams
                except:
                    print("Could not fit curve to PtI data (yet)")
                try:
                    self.itpCurveParams, itpCurveCovar = optimize.curve_fit(FcurveFunc, flowData, itpData, maxfev=1000)
                    timeDelayFCurves.loc[pressure,'ItP']=self.itpCurveParams
                except:
                    print("Could not fit curve to ItP data (yet)")
            plotPanel.updateDelayCurves()
    
    def fitPFfunctions(self):    
        axXpar = 'Pressure'
        axXunit = 'mbar'
        axYpar = 'Flow'
        axYunit = 'mln/min'
        
        PtIdata = []
        ItPdata = []
        for index, row in timeDelayData.iterrows():
            PtIdataPoint = [row[axXpar],row[axYpar],row['PtI']]
            ItPdataPoint = [row[axXpar],row[axYpar],row['ItP']]
            PtIdata=PtIdata + [PtIdataPoint]
            ItPdata=ItPdata + [ItPdataPoint]
        
        PtIx_data = []
        PtIy_data = []
        PtIz_data = []
        for item in PtIdata:
            PtIx_data.append(item[0])
            PtIy_data.append(item[1])
            PtIz_data.append(item[2])
            
        ItPx_data = []
        ItPy_data = []
        ItPz_data = []
        for item in ItPdata:
            ItPx_data.append(item[0])
            ItPy_data.append(item[1])
            ItPz_data.append(item[2])
        
        # Fit the function to the datasets
        PtIparameters, PtIcovariance = optimize.curve_fit(PFcurveFunc, [PtIx_data, PtIy_data], PtIz_data)
        ItPparameters, ItPcovariance = optimize.curve_fit(PFcurveFunc, [ItPx_data, ItPy_data], ItPz_data)
        
        timeDelayPFCurves['PtI']=[PtIparameters]
        timeDelayPFCurves['ItP']=[ItPparameters]
        timeDelayPFCurves['Type']='3D'
       
        PtImodel_x_data = np.linspace(min(PtIx_data), max(PtIx_data), 30)
        PtImodel_y_data = np.linspace(min(PtIy_data), max(PtIy_data), 30)
        ItPmodel_x_data = np.linspace(min(ItPx_data), max(ItPx_data), 30)
        ItPmodel_y_data = np.linspace(min(ItPy_data), max(ItPy_data), 30)
        
        PtIX, PtIY = np.meshgrid(PtImodel_x_data, PtImodel_y_data)
        ItPX, ItPY = np.meshgrid(ItPmodel_x_data, ItPmodel_y_data)

        PtIZ = PFcurveFunc(np.array([PtIX, PtIY]), *PtIparameters)
        ItPZ = PFcurveFunc(np.array([ItPX, ItPY]), *ItPparameters)
        
        fig = plt.figure(figsize=plt.figaspect(0.5))
        
        # PtI subplot
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.view_init(elev=30., azim=135)
        ax.set_title('PtI Delay Function')
        # plot surface
        ax.plot_surface(PtIX, PtIY, PtIZ, cmap="viridis_r", alpha=0.2)
        ax.plot_wireframe(PtIX,PtIY,PtIZ, cmap="viridis_r", linewidth=1)
        
        for index in range(len(PtIx_data)):
            x=PtIx_data[index]
            y=PtIy_data[index]
            z=PtIz_data[index]
            zs = PFcurveFunc([x,y],PtIparameters[0],PtIparameters[1],PtIparameters[2],PtIparameters[3],PtIparameters[4],PtIparameters[5])
            ax.plot([x,x],[y,y],[z,zs],color='red')
            ax.plot(x,y,zs,marker="x", color='red')
            
        ax.scatter(PtIx_data, PtIy_data, PtIz_data, depthshade=False, alpha = 1, color='darkgreen')
            
        ax.set_xlabel(axXpar + " ("+axXunit+")")
        ax.set_ylabel(axYpar + " ("+axYunit+")")
        ax.set_zlabel("PtI delay (s)")
        
        #---- ItP subplot
        ax = fig.add_subplot(1, 2, 2, projection='3d')
        ax.view_init(elev=30., azim=135)
        ax.set_title('ItP Delay Function')
        # plot surface
        ax.plot_surface(ItPX, ItPY, ItPZ, cmap="viridis_r", alpha=0.2)
        ax.plot_wireframe(ItPX,ItPY,ItPZ, cmap="viridis_r", linewidth=1)
        
        for index in range(len(ItPx_data)):
            x=ItPx_data[index]
            y=ItPy_data[index]
            z=ItPz_data[index]
            zs = PFcurveFunc([x,y],ItPparameters[0],ItPparameters[1],ItPparameters[2],ItPparameters[3],ItPparameters[4],ItPparameters[5])
            ax.plot([x,x],[y,y],[z,zs],color='red')
            ax.plot(x,y,zs,marker="x", color='red')
            
        ax.scatter(ItPx_data, ItPy_data, ItPz_data, depthshade=False, alpha = 1, color='darkgreen')
        
        ax.set_xlabel(axXpar + " ("+axXunit+")")
        ax.set_ylabel(axYpar + " ("+axYunit+")")
        ax.set_zlabel("ItP delay (s)")
        
        plt.show()
    
    def setPressureConditions(self):
        newState = self.sequence.iloc[self.sequenceStep]
        pressure = newState['P']
        pressureOffset = newState['PO']
        self.currentInletPressure, self.currentOutletPressure = calcInletOutletPressures(pressure,pressureOffset)
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
            if self.dataCollectors[0].processedData.tail(self.dataCollectors[0].parInfo['minStableDuration'])['absDiffSum'].max() < self.dataCollectors[0].parInfo['stableThreshold']:
                if self.dataCollectors[1].processedData.tail(self.dataCollectors[1].parInfo['minStableDuration'])['absDiffSum'].max() < self.dataCollectors[1].parInfo['stableThreshold']:
                    if self.dataCollectors[2].processedData[self.dataCollectors[2].processedData['absDiffSum'].notna()].tail(self.dataCollectors[2].parInfo['minStableDuration'])['absDiffSum'].max() < self.dataCollectors[2].parInfo['stableThreshold']:
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
    
                   
    def initiateChange(self):
        print("Initiating gas change")
        plotPanel.setStatus(self.sequenceStep+1,self.sequence.shape[0],"Initiating gas change")
        timeDelayData.at[self.sequenceStep,'Pressure']=self.sequence.iloc[self.sequenceStep]['P']
        timeDelayData.at[self.sequenceStep,'PressureOffset']=self.sequence.iloc[self.sequenceStep]['PO']
        timeDelayData.at[self.sequenceStep,'It']=self.sequence.iloc[self.sequenceStep]['It']
        initiateTime = impulse.gas.data.getNewData()[timePar]
        self.initiateTime = initiateTime
        self.lastDetectionTime = initiateTime
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


    def findChangeStart(self, dataCollector, initTime, changeTime):
        parInfo = dataCollector.parInfo
        beforeBuffer = 3 #seconds
        afterBuffer = 3 #seconds
        dataRange = dataCollector.processedData[(dataCollector.processedData[timePar].gt(initTime-beforeBuffer)) & (dataCollector.processedData[timePar].lt(changeTime+afterBuffer))]
        dataRange = dataRange[dataRange['absDiffSum'].notna()]
        dataRange['changing']=dataRange['absDiffSum'].gt(parInfo['stableThreshold'])
        dataRange['changeStartStop'] = (dataRange['changing'] == dataRange['changing'].shift(1))
        changeTimes = dataRange[dataRange['changeStartStop']==False]
        changeTimes = changeTimes.iloc[1: , :] #Drop the first changeTime, the first row always returns False
        changeTimes = changeTimes[changeTimes['changing']==True]

        if changeTimes.shape[0]>0:
            changeTime = changeTimes.iloc[-1][timePar]
            return changeTime
        else:
            print("no change found")
            return 0


    def detectChange(self):
        dataCollector = self.dataCollectors[self.checkChangePos]
        locations = ["pre-TEM", "in-TEM", "post-TEM"]
        print(f"Detecting change {locations[self.checkChangePos]}")
        plotPanel.setStatus(self.sequenceStep+1,self.sequence.shape[0],f"Detecting change {locations[self.checkChangePos]}")
        lastFound = 0
        dataSinceInit = dataCollector.processedData[dataCollector.processedData[timePar]>self.initiateTime]
        tresFilt = dataSinceInit[dataSinceInit['absDiffSum'].gt(dataCollector.parInfo['changeThreshold'])]
        if tresFilt.shape[0]>0:
            changeDetectTime=tresFilt.iloc[-1][timePar]
            startChangeTime = self.findChangeStart(dataCollector, self.lastDetectionTime, changeDetectTime)
           
            if startChangeTime!=0:
                self.lastDetectionTime = startChangeTime
                timeDelayData.at[self.sequenceStep,self.checkChangePositions[self.checkChangePos]]=startChangeTime
                flagName = 'P'+str(self.sequence.iloc[self.sequenceStep]['P'])+'O'+str(self.sequence.iloc[self.sequenceStep]['PO'])+'#'+str(self.sequence.iloc[self.sequenceStep]['It'])
                timeDelayData.at[self.sequenceStep,'Cid']=flagName

                if self.checkChangePos == 2:
                    dataForMeanFlow = self.dataCollectors[0].processedData[self.dataCollectors[0].processedData[timePar]>=self.initiateTime]
                    meanFlow = dataForMeanFlow['reactorFlowMeasured'].mean()
                    timeDelayData.at[self.sequenceStep,'Flow']=meanFlow
                    timeDelayData.at[self.sequenceStep,'PtI']=timeDelayData.loc[self.sequenceStep,'inChangeTime']-timeDelayData.loc[self.sequenceStep,'preChangeTime']
                    timeDelayData.at[self.sequenceStep,'ItP']=timeDelayData.loc[self.sequenceStep,'postChangeTime']-timeDelayData.loc[self.sequenceStep,'inChangeTime']
                    self.fitCurves()
                    lastFound = 1
                else:
                    print(f"Found change {self.checkChangePos} at time {startChangeTime}")
                    self.checkChangePos +=1     
        return lastFound

    
    def saveData(self):
        now = datetime.now()
        dt_string = now.strftime("%d-%m-%Y_%H-%M-%S_")
        #timeDelayData.to_csv(dt_string + 'timeDelayData.csv')
        timeDelayPFCurves.to_csv(dt_string + 'timeDelayCalibration.csv')
    
    
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
            self.detectChangeTimeoutCounter=0
            self.state += self.initiateChange()      
        elif self.state == 4: 
            self.state += self.detectChange()
            self.detectChangeTimeoutCounter +=1
            if self.detectChangeTimeoutVal - self.detectChangeTimeoutCounter < 10:
                print(f"Timeout in {self.detectChangeTimeoutVal - self.detectChangeTimeoutCounter}")
                plotPanel.setStatus(self.sequenceStep+1,self.sequence.shape[0],f"Timeout in {self.detectChangeTimeoutVal - self.detectChangeTimeoutCounter}")
                if self.detectChangeTimeoutVal == self.detectChangeTimeoutCounter:
                    print("Change not detected before timeout, restarting measurement")
                    plotPanel.setStatus(self.sequenceStep+1,self.sequence.shape[0],"Change not detected before timeout, restarting measurement")
                    self.state = 0
        elif self.state == 5:
            if self.sequenceStep+1 > self.sequence.shape[0]-1:
                print("Last test finished")
                plotPanel.setStatus(self.sequenceStep+1,self.sequence.shape[0],"Last test finished.")
                self.testActive = False
            elif self.sequenceStep+1 < self.sequence.shape[0]:
                if self.sequence.iloc[self.sequenceStep+1]['nPO']==1: #If the next test will be at new pressures, no need to wait for stable conditions
                    self.state+=1
                else: self.state += self.waitUntilStable()     
        elif self.state == 6: # Update sequence number
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
controller.fitPFfunctions()
controller.saveData() # Save the delay data to a file
impulse.disconnect() # Close the connection with Impulse