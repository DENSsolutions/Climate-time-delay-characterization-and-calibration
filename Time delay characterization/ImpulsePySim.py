"""
ImpulsePy simulator for Time Delay Curve Characterization Script
Version 1.0
Thu Jul 15 16:25:22 2021

@author: Merijn Pen
"""

import datetime as dt
from time import sleep
import pandas as pd
import random
import math

initTime = dt.datetime.now() - dt.timedelta(seconds=200)

def ptiDelayFunction(x):
    a=-0.00537
    b=0.19851
    c=-0.3538
    d=0.36659
    return 1/(a+b*x+c*x**2+d*x**3)

def itpDelayFunction(x):
    a=-0.00767
    b=0.01736
    c=0.66376
    d=-0.44175
    return 1/(a+b*x+c*x**2+d*x**3)+ptiDelayFunction(x)

def pressureDependency(pressure):
    return pressure/400

def waitForControl():
    sleep(1)

def disconnect():
    print("Bye.")

class gasDataGenerator():
    def __init__(self, interval, initTime):
        self.dataSet = pd.DataFrame()
        self.setPointData = pd.DataFrame()
        self.initTime = initTime
        self.flags = {}
        initialSetpoint = {
            "experimentDuration" : 0,
            'inletPressureSetpoint' : 0,
            'outletPressureSetpoint' : 0,
            'gas1FlowSetpoint' : 0,
            'gas3FlowSetpoint' : 0
            }
        self.setPointData = self.setPointData.append(initialSetpoint, ignore_index=True)
        self.interval = interval
        self.sequenceNumber = 0
        self.lastData = None
        self.currentFlag = ""
    
    def subscribe(self):
        print("Subscribed to fake gasData")

    def calculateNext(self, init, newTimestamp, parametername, setpointname, delay, speed):
        if self.dataSet.shape[0]<1:
            return init
        else:
            if self.setPointData.shape[0]>0:
                tresHold = float(newTimestamp-delay)
                availableSetpoints = self.setPointData[self.setPointData['experimentDuration']<tresHold]               
                if availableSetpoints.shape[0]>0: targetVal = availableSetpoints.iloc[-1][setpointname]
                else: targetVal = init
            else: targetVal = init
            lastVal = self.dataSet.iloc[-1][parametername]
            newVal = lastVal + ((targetVal - lastVal) * speed)
            newVal = newVal + (((random.random()-0.5)/500))
            return newVal

    def calculateFlow(self, inlet, outlet):
        offset = inlet-outlet
        flow = offset*0.00055
        return flow
    
    def generateNewData(self):
        if self.dataSet.shape[0]<1:
            lastTimeStamp = 0
        else: lastTimeStamp = self.dataSet.iloc[-1]['experimentDuration']
        newTimeStamp = lastTimeStamp + self.interval
        while self.initTime + dt.timedelta(seconds = newTimeStamp) < dt.datetime.now():
            nextInlet = self.calculateNext(0, newTimeStamp, 'inletPressureMeasured', 'inletPressureSetpoint', 1 ,0.5)
            nextOutlet = self.calculateNext(0, newTimeStamp ,'outletPressureMeasured', 'outletPressureSetpoint', 1 ,0.5)
            nextFlow = self.calculateFlow(nextInlet, nextOutlet)
            nextGas1Flow = self.calculateNext(0, newTimeStamp, 'gas1FlowMeasured', 'gas1FlowSetpoint', 1, 0.3)
            dataMsg = {
                "sequenceNumber" : self.sequenceNumber,
                'experimentDuration' : newTimeStamp,                   
                'inletPressureSetpoint' : self.setPointData.iloc[-1]['inletPressureSetpoint'],
                'outletPressureSetpoint' : self.setPointData.iloc[-1]['outletPressureSetpoint'],
                'gas1FlowSetpoint' : self.setPointData.iloc[-1]['gas1FlowSetpoint'],
                'gas3FlowSetpoint' : self.setPointData.iloc[-1]['gas3FlowSetpoint'],
                'inletPressureMeasured' : nextInlet,
                'outletPressureMeasured' : nextOutlet,
                'reactorFlowMeasured' : nextFlow,
                'gas1FlowMeasured' : nextGas1Flow
                }

            self.sequenceNumber += 1
            self.lastData=dataMsg
            self.dataSet = self.dataSet.append(dataMsg, ignore_index=True)
            lastTimeStamp = newTimeStamp
            newTimeStamp = lastTimeStamp + self.interval

    def getNewData(self):
        self.generateNewData()
        return self.lastData

    def getLastData(self):
        self.generateNewData()
        return self.lastData

    def setFlag(self, flagName):
        self.flags[flagName]=self.dataSet.shape[0]
        self.currentFlag = flagName
        
    def getDataFrame(self, startRowOrFlag=1, endRowOrFlag=None):
        self.generateNewData()
        if isinstance(startRowOrFlag, str):
            if startRowOrFlag in self.flags:
                startRowOrFlag= self.flags[startRowOrFlag]
            else:
                print(f"[ERROR] Flag {startRowOrFlag} does not exist!")
                return []
        if isinstance(endRowOrFlag, str):
            if endRowOrFlag in self.flags:
                endRowOrFlag= self.flags[endRowOrFlag]
            else:
                print(f"[ERROR] Flag {endRowOrFlag} does not exist!")
                return []
        if startRowOrFlag: startRowOrFlag = int(startRowOrFlag)
        if endRowOrFlag: endRowOrFlag = int(endRowOrFlag)
        df = self.dataSet.iloc[startRowOrFlag:endRowOrFlag].copy()
        return df


class heatDataGenerator():
    def __init__(self, interval, initTime, gasData):
        self.dataSet = pd.DataFrame()
        self.gasData = gasData
        self.setPointData = pd.DataFrame()
        self.initTime = initTime
        self.flags = {}
        self.currentFlag = ""
        initialSetpoint = {
            "experimentDuration" : 0,
            'setpoint' : 21     
            }
        self.setPointData = self.setPointData.append(initialSetpoint, ignore_index=True)
        self.interval = interval
        self.sequenceNumber = 0
        self.lastData = None
    
    def subscribe(self):
        print("Subscribed to fake heatData")  

    def temperatureMeasured(self):
        if self.dataSet.shape[0]<1:
            return 25
        else:
            targetVal = self.setPointData.iloc[-1]['setpoint']
            lastVal = self.dataSet.iloc[-1]['temperatureMeasured']
            newVal = lastVal + ((targetVal - lastVal) * 0.3)
            return newVal   
    
    def powerMeasured(self, temperatureMeasured, newTimeStamp):
        if self.dataSet.shape[0]<1:
            return 0.3
        else:
            lastVal = self.dataSet.iloc[-1]['powerMeasured']
            flowSpeed = self.gasData.dataSet.iloc[-1]['reactorFlowMeasured']

            Pin = self.gasData.dataSet.iloc[-1]['inletPressureMeasured']
            Pout = self.gasData.dataSet.iloc[-1]['outletPressureMeasured']
            pressure =  math.sqrt((0.5*(Pin**2))+(0.5*(Pout**2)))
            timeDelay = ptiDelayFunction(flowSpeed)+pressureDependency(pressure)
            print(f"{timeDelay} = {ptiDelayFunction(flowSpeed)} + {pressureDependency(pressure)}")
            
            gasData = self.gasData.dataSet[self.gasData.dataSet['experimentDuration']<(newTimeStamp-timeDelay)]
            if gasData.shape[0]>0:
                target = self.gasData.dataSet[self.gasData.dataSet['experimentDuration']<(newTimeStamp-timeDelay)].iloc[-1]['gas1FlowMeasured']
            else:  target = lastVal
            targetPoint = (temperatureMeasured/1000) - (target / 8)
            newVal = lastVal + ((targetPoint - lastVal) * 0.3)
            newVal = newVal + (((random.random()-0.5)/10000))
            return newVal   
    
    def generateNewData(self):
        if self.dataSet.shape[0]<1:
            lastTimeStamp = 0
        else: lastTimeStamp = self.dataSet.iloc[-1]['experimentDuration']
        newTimeStamp = lastTimeStamp + self.interval
        while self.initTime + dt.timedelta(seconds = newTimeStamp) < dt.datetime.now():
            
            temperatureMeasured = self.temperatureMeasured()
            powerMeasured = self.powerMeasured(temperatureMeasured, newTimeStamp)
            
            dataMsg = {
                "sequenceNumber" : self.sequenceNumber,
                'experimentDuration' : newTimeStamp,                   
                'temperatureSetpoint' : self.setPointData.iloc[-1]['setpoint'],
                'temperatureMeasured' : temperatureMeasured,
                'powerMeasured' : powerMeasured
                }

            self.sequenceNumber += 1
            self.lastData=dataMsg
            self.dataSet = self.dataSet.append(dataMsg, ignore_index=True)
            lastTimeStamp = newTimeStamp
            newTimeStamp = lastTimeStamp + self.interval
    
    def getNewData(self):
        self.generateNewData()
        return self.lastData

    def getLastData(self):
        self.generateNewData()
        return self.lastData
        
    def setFlag(self, flagName):
        self.flags[flagName]=self.dataSet.shape[0]
        self.currentFlag = flagName
        
    def getDataFrame(self, startRowOrFlag=1, endRowOrFlag=None):
        self.generateNewData()
        if isinstance(startRowOrFlag, str):
            if startRowOrFlag in self.flags:
                startRowOrFlag= self.flags[startRowOrFlag]
            else:
                print(f"[ERROR] Flag {startRowOrFlag} does not exist!")
                return []
        if isinstance(endRowOrFlag, str):
            if endRowOrFlag in self.flags:
                endRowOrFlag= self.flags[endRowOrFlag]
            else:
                print(f"[ERROR] Flag {endRowOrFlag} does not exist!")
                return []
        if startRowOrFlag: startRowOrFlag = int(startRowOrFlag)
        if endRowOrFlag: endRowOrFlag = int(endRowOrFlag)
        df = self.dataSet.iloc[startRowOrFlag:endRowOrFlag].copy()
        return df

class msDataGenerator():
    def __init__(self, interval, initTime, gasData):
        self.dataSet = pd.DataFrame()
        self.gasData = gasData
        self.flags = {}
        self.currentFlag = ""
        self.initTime = initTime
        self.interval = interval
        self.sequenceNumber = 0
        self.lastData = None
    
    def subscribe(self):
        print("Subscribed to fake msData")
    
    def calculateNext(self, newTimeStamp):
        if self.dataSet.shape[0]<1:
            return 1E-7
        else:
            lastVal = self.dataSet.iloc[-1]['Methane']
            flowSpeed = self.gasData.dataSet.iloc[-1]['reactorFlowMeasured']
            Pin = self.gasData.dataSet.iloc[-1]['inletPressureMeasured']
            Pout = self.gasData.dataSet.iloc[-1]['outletPressureMeasured']
            pressure =  math.sqrt((0.5*(Pin**2))+(0.5*(Pout**2)))
            print(f"3: {pressure}")
            timeDelay = itpDelayFunction(flowSpeed)+(pressureDependency(pressure)*2)
            gasData = self.gasData.dataSet[self.gasData.dataSet['experimentDuration']<(newTimeStamp-timeDelay)]
            if gasData.shape[0]>0:
                target = self.gasData.dataSet[self.gasData.dataSet['experimentDuration']<(newTimeStamp-timeDelay)].iloc[-1]['gas1FlowMeasured']
            else: target = lastVal
            targetPoint = target / 5000000
            newVal = lastVal + ((targetPoint - lastVal) * 0.01)
            newVal = newVal + (((random.random()-0.5)/5000000000))
            return newVal   
    
    def generateNewData(self):
        if self.dataSet.shape[0]<1:
            lastTimeStamp = 0
        else: lastTimeStamp = self.dataSet.iloc[-1]['experimentDuration']
        newTimeStamp = lastTimeStamp + self.interval
        while self.initTime + dt.timedelta(seconds = newTimeStamp) < dt.datetime.now():
            dataMsg = {
                "sequenceNumber" : self.sequenceNumber,
                'experimentDuration' : newTimeStamp,                   
                'Methane' : self.calculateNext(newTimeStamp),
                }

            self.sequenceNumber += 1
            self.lastData=dataMsg
            self.dataSet = self.dataSet.append(dataMsg, ignore_index=True)
            lastTimeStamp = newTimeStamp
            newTimeStamp = lastTimeStamp + self.interval

    def getNewData(self):
        self.generateNewData()
        return self.lastData

    def getLastData(self):
        self.generateNewData()
        return self.lastData
        
    def setFlag(self, flagName):
        self.flags[flagName]=self.dataSet.shape[0]
        self.currentFlag = flagName
        
    def getDataFrame(self, startRowOrFlag=1, endRowOrFlag=None):
        self.generateNewData()
        if isinstance(startRowOrFlag, str):
            if startRowOrFlag in self.flags:
                startRowOrFlag= self.flags[startRowOrFlag]
            else:
                print(f"[ERROR] Flag {startRowOrFlag} does not exist!")
                return []
        if isinstance(endRowOrFlag, str):
            if endRowOrFlag in self.flags:
                endRowOrFlag= self.flags[endRowOrFlag]
            else:
                print(f"[ERROR] Flag {endRowOrFlag} does not exist!")
                return []
        if startRowOrFlag: startRowOrFlag = int(startRowOrFlag)
        if endRowOrFlag: endRowOrFlag = int(endRowOrFlag)
        df = self.dataSet.iloc[startRowOrFlag:endRowOrFlag].copy()
        return df


gasData = gasDataGenerator(1, initTime)
heatData = heatDataGenerator(0.3, initTime, gasData)
msData = msDataGenerator(0.2, initTime, gasData)


class gasControlClass():
    def __init__(self, gasDataGenerator, msDataGenerator):
        self.data = gasDataGenerator
        self.msdata = msDataGenerator
        
    def setIOP(self, inletPressureSetpoint, outletPressureSetpoint, gas1FlowSetpoint, gas1FlowDirection, gas2FlowSetpoint, gas2FlowDirection, gas3FlowSetpoint, gas3FlowDirection):
        experimentTime = dt.datetime.now()-initTime
        point = {
            "experimentDuration" : experimentTime.total_seconds(),
            'inletPressureSetpoint' : inletPressureSetpoint,
            'outletPressureSetpoint' : outletPressureSetpoint,
            'gas1FlowSetpoint' : gas1FlowSetpoint
            }
        self.data.setPointData = self.data.setPointData.append(point, ignore_index=True)


class heatControlClass():
    def __init__(self, dataGenerator):
        self.data = dataGenerator
    
    def set(self, setpoint):
        point = {
            "experimentDuration" : dt.datetime.now().strftime('%H:%M:%S.%f')[:-3],
            'setpoint' : setpoint           
            }
        self.data.setPointData = self.data.setPointData.append(point, ignore_index=True)


heat = heatControlClass(heatData)
gas = gasControlClass(gasData, msData)

