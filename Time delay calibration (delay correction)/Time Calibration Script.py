import pandas as pd
import numpy as np
import math
import os
from pathlib import Path
import scipy.interpolate

#############################################
#                                           #
#  Set your directories in the lines below  #
#                                           #
#############################################

impulseLogfilePath = Path("Data/FullTimeDelayScriptTest_Synchronized data.csv") # Change this location to the location of your impulse logfile
MSLogfilePath =  "" # Leave empty if MS data is in the impulse logfile
timeDelayCalibrationPath = Path("Data/10-08-2021_15-28-50_timeDelayCurvePars.csv") # Default curve parameters are used when left empty

#############################################

beforeTemParameters = ["TimeStamp", "Experiment time", "MFC1 Measured", "MFC1 Setpoint","MFC2 Measured", "MFC2 Setpoint","MFC3 Measured", "MFC3 Setpoint", "MixValve", "% Gas1 Measured", "% Gas2 Measured", "% Gas3 Measured", "% Gas1 Setpoint", "% Gas2 Setpoint", "PumpRotation",  "ActiveProgram"]
inTemParameters = ["TimeStamp", "Experiment time", "Fnr", "Fnr Setpoint", "Temperature Setpoint","Temperature Measured", "Pin Measured", "Pin Setpoint", "Pout Measured", "Pout Setpoint", "Pnr (Calculated from Pin Pout)", "Pnr Setpoint","Measured power", "Pvac", "Relative power reference", "Relative power"]
afterTemParameters = ["TimeStamp", "Experiment time", "Channel#1", "Channel#2", "Channel#3", "Channel#4", "Channel#5", "Channel#6", "Channel#7", "Channel#8", "Channel#9", "Channel#10"]

if timeDelayCalibrationPath!="":
    print("Loaded curve parameters used.")
    curveParameters = pd.read_csv(timeDelayCalibrationPath)
    beforeCurveParameters = curveParameters['PtI'][0]
    afterCurveParameters = curveParameters['ItP'][0]
    while "  " in beforeCurveParameters: beforeCurveParameters = beforeCurveParameters.replace("  ", " ")
    while "  " in afterCurveParameters: afterCurveParameters = afterCurveParameters.replace("  ", " ")
    beforeCurveParameters = [float(i) for i in beforeCurveParameters.strip("[]").strip().split(" ")]
    afterCurveParameters = [float(i) for i in afterCurveParameters.strip("[]").strip().split(" ")]
    
else:
    print("Standard curve parameters used.")
    beforeCurveParameters = [-0.00365, -0.14249, 0.09555]
    afterCurveParameters = [0.00189, 0.04113, 0.09746]

def calculateBeforeOffset(flowrate): # Calculates the time delay between before-TEM and inside-TEM
    a = beforeCurveParameters[0]
    b = beforeCurveParameters[1]
    c = beforeCurveParameters[2]
    return a * np.exp(-b * flowrate) + c

def calculateAfterOffset(flowrate): # Calculates the time delay between inside-TEM and after-TEM
    a = afterCurveParameters[0]
    b = afterCurveParameters[1]
    c = afterCurveParameters[2]
    return a * np.exp(-b * flowrate) + c

#Load the Impulse logfile into a pandas dataframe
allData = pd.read_csv(impulseLogfilePath, infer_datetime_format=True)
allData['TimeStamp']= pd.to_datetime(allData['TimeStamp']).dt.time
    
#Separate parameters into before in and after TEM data
beforeTemData = allData.filter(items=beforeTemParameters)
inTemData = allData.filter(items=inTemParameters)
afterTemData = None
if 'Channel#1' in allData: afterTemData = allData.filter(items=afterTemParameters)
    
#If there is a separate MS logfile, load that one (which is a messy business)
if MSLogfilePath!="": 
    headerItems = ["Experiment time"]
    channelSection = 0
    lineCounter = 0
    with open(MSLogfilePath, 'r') as csvfile:
        for line in csvfile:
            line = line.strip()
            if line[:10]=="Start time":
                timeString = line[26:-3]
                timeString = timeString + ".000"
                MSstartTime=datetime.strptime(timeString,"%H:%M:%S.%f")
                if line[-2:]=="PM":
                    MSstartTime = MSstartTime + timedelta(hours=12)
            if line[:7]=="Time(s)":
                channelSection = 0
                headerLength = lineCounter
            if channelSection == 1:
                while '  ' in line:
                    line = line.replace('  ',',')
                while ',,' in line:
                    line = line.replace(',,',',')            
                line = line.split(',')
                if len(line) > 1:    
                    headerItems.append(line[2])
            if line[:7]=="Channel":
                channelSection = 1
            lineCounter=lineCounter+1
    afterTemData = pd.read_csv(MSLogfilePath, header=0, index_col=False, names=headerItems, skiprows=headerLength)
    
    #Calculate the true Impulse starttime (first timestamp - first experiment time seconds)
    impulseExpTimStartMil, impulseExpTimStartSec = math.modf(allData['Experiment time'].iloc[0])
    impulseExpTimStartMil = round(impulseExpTimStartMil,3)*1000
    impulseFirstTimeStamp = datetime.combine(date.today(),allData['TimeStamp'].iloc[0])
    realImpulseStartTime = impulseFirstTimeStamp - timedelta(seconds=impulseExpTimStartSec, milliseconds=impulseExpTimStartMil)
    
    # Calculate the number of seconds offset between the experiment time of the MS and the experiment time of Impulse
    if MSstartTime > realImpulseStartTime:
        experimentTimeOffset = (MSstartTime - realImpulseStartTime).seconds + ((MSstartTime - realImpulseStartTime).microseconds/1000000)
    else:
        experimentTimeOffset = -((realImpulseStartTime-MSstartTime).seconds + ((realImpulseStartTime-MSstartTime).microseconds/1000000))
    
    
    # Calculate the MS TimeStamps based on MSstartTime and the experiment time, and adjust the experiment time with the offset from Impulse
    afterTemData.insert(0,'StartTime',MSstartTime)
    afterTemData.insert(0,'Experimentsec','')
    afterTemData.insert(0,'TimeStamp','')
    afterTemData['Experimentsec']=pd.to_timedelta(afterTemData['Experiment time'] ,'s')
    afterTemData['TimeStamp']=(afterTemData['StartTime']+afterTemData['Experimentsec']).dt.time
    del afterTemData['StartTime']
    del afterTemData['Experimentsec']
    afterTemData['Experiment time']=afterTemData['Experiment time']+experimentTimeOffset
    
#Calculate rolling average for Flow to prevent sudden changes from timewarping the data
RAwindow=5
inTemData['Fnr RA'] = inTemData['Fnr'].rolling(window=RAwindow,center=True).mean()
inTemData['Fnr RA'].fillna(inTemData['Fnr'], inplace=True) #Fill the missing Fnr RA values at the head and tail with the original values

#Correct beforeTemData
beforeTemDataCorrected = beforeTemData.copy()
beforeTemDataCorrected['Fnr RA']=inTemData['Fnr RA']
beforeTemDataCorrected['Time correction (seconds)']=np.vectorize(calculateBeforeOffset)(beforeTemDataCorrected['Fnr RA'])
beforeTemDataCorrected['Time correction timedelta']= pd.to_timedelta(beforeTemDataCorrected['Time correction (seconds)'] ,'s')
beforeTemDataCorrected['TimeStamp']= (pd.to_datetime(beforeTemDataCorrected['TimeStamp'].astype(str))+beforeTemDataCorrected['Time correction timedelta']).dt.time
beforeTemDataCorrected['Experiment time']+=beforeTemDataCorrected['Time correction (seconds)']
del beforeTemDataCorrected['Time correction timedelta']
del beforeTemDataCorrected['Fnr RA']

#Correct afterTemData
if afterTemData is not None:
    afterTemDataCorrected = afterTemData.copy()
    if MSLogfilePath=="": #If the MS data was included in the Impulse logfile (same timestamps)
        afterTemDataCorrected['Fnr RA']=inTemData['Fnr RA']

    if MSLogfilePath!="": #Different logfile for MS, so Fnr RA has to be interpolated
        # Interpolate Fnr RA to calculate offsets for MS data
        FnrRAInterp = scipy.interpolate.interp1d(inTemData['Experiment time'],inTemData['Fnr RA'])

        #Crop MS logfile so that Experiment time values fall within interpolated range
        minTime = inTemData['Experiment time'].iloc[0]
        maxTime = inTemData['Experiment time'].iloc[-1]
        afterTemDataCorrected=afterTemDataCorrected[afterTemDataCorrected['Experiment time']>minTime]
        afterTemDataCorrected=afterTemDataCorrected[afterTemDataCorrected['Experiment time']<maxTime]

        #Find the Fnr RA values for the MS timestamps
        afterTemDataCorrected['Fnr RA']=np.vectorize(FnrRAInterp)(afterTemDataCorrected['Experiment time'])

    afterTemDataCorrected['Time correction (seconds)']=np.vectorize(calculateAfterOffset)(afterTemDataCorrected['Fnr RA'])
    afterTemDataCorrected['Time correction timedelta']= pd.to_timedelta(afterTemDataCorrected['Time correction (seconds)'] ,'s')
    afterTemDataCorrected['TimeStamp']= (pd.to_datetime(afterTemDataCorrected['TimeStamp'].astype(str))-afterTemDataCorrected['Time correction timedelta']).dt.time
    afterTemDataCorrected['Experiment time']-=afterTemDataCorrected['Time correction (seconds)']
    del afterTemDataCorrected['Time correction timedelta']
    del afterTemDataCorrected['Fnr RA']

#Function to format the timestamp the way we like it (with only 3 digits for ms)
def timeStampFormatter(dt):
    return "%s:%.3f%s" % (
        dt.strftime('%H:%M'),
        float("%.3f" % (dt.second + dt.microsecond / 1e6)),
        dt.strftime('%z')
    )

#Run the timeStampFormatter for the after and before datasets
if afterTemData is not None: afterTemDataCorrected['TimeStamp']= np.vectorize(timeStampFormatter)(afterTemDataCorrected['TimeStamp'])
beforeTemDataCorrected['TimeStamp']= np.vectorize(timeStampFormatter)(beforeTemDataCorrected['TimeStamp'])

# Paths and filenames
experimentName = os.path.splitext(impulseLogfilePath.name)[0]
correctedDataFolder = os.path.dirname(impulseLogfilePath)#+"/"+experimentName+"_corrected-data"
Path(correctedDataFolder).mkdir(parents=True, exist_ok=True) # Create corrected-data folder

#Create the CSV files
inTemData.to_csv((correctedDataFolder+'/'+experimentName+'_corrected-inside.csv'), index=False)
beforeTemData.to_csv((correctedDataFolder+'/'+experimentName+'_corrected-before.csv'), index=False)            
if afterTemData is not None: afterTemData.to_csv((correctedDataFolder+'/'+experimentName+'_corrected-after.csv'), index=False)
   
beforeTemDataCorrected = beforeTemDataCorrected.sort_values(by = 'Experiment time')
inTemData = inTemData.sort_values(by = 'Experiment time')
syncData = pd.merge_asof(inTemData, beforeTemDataCorrected, on = 'Experiment time')
if afterTemData is not None:
    afterTemDataCorrected = afterTemDataCorrected.sort_values(by = 'Experiment time')
    syncData = pd.merge_asof(syncData, afterTemDataCorrected, on = 'Experiment time')

syncData.to_csv((correctedDataFolder+'/'+experimentName+'_corrected-synchronized.csv'), index=False)