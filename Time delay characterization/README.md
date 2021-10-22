# Climate time delay characterization script
This repository contains the following scripts:
- Time Delay Curve Characterization script

>The time delay characterization script automatically measures the time delay between the pre-TEM, in-TEM and post-TEM measurement locations at different flow-speeds.
>This script is writted based on the results of the collaboration between DENSsolutions and DICP.

- ImpulsePySim
>This is a script that simulates measurements with time delays for the Time delay curve characterization script. To run the characterization script with this simulator, make sure to disable the importing of the impulsePy module and enable the line below it.

- Fit 3D function
>This script can load the timedelaydata.csv file that is saved by the time delay curve characterization script and will fit a 3D function to the data. Eventually the Time delay Calibration script will be updated to use this 3D function for the correction of the data.


## Dependencies
In order to run the time delay characterization script, you will need to have the impulsePy module installed.
ImpulsePy can be installed using: 
```bash
pip install impulsePy
```

Additionally, this script uses the following modules:
- scipy
- pandas
- numpy
- seaborn


## Script configurations
By default, the script is configured to monitor the Measured gas 1 flowspeed for the pre-TEM signal, the measured heater power for the in-TEM signal and the Methane RGA signal for the post-TEM signal.

In the code section "Parameters", you can specify a few test settings:
- The temperature at which the measurements are taken
- The reactor-pressure at which the measurements are taken (it is possible to do multiple tests at different reactor pressures by adding them to this list)
- The different pressure-offsets at which the script will take time delay measurements, these are the offsets between the Inlet pressure and the Outlet pressure, calculated as follows: Pin=Reactor pressure+0.5*Pressure Offset
- The number of iterations/measurements at each pressure offset
- The two gas states, gas state A and B, between which the script will toggle in order to measure the time delay
- Three dicts define settings for the pre-TEM, in-TEM and post-TEM parameter, such as the parameter name (should correspond with the name in the API communication), the size of the rolling average window, the thresholds for the stable and changing states and the minimum stable duration.

## Tweaking the configurations for your system
Sometimes, the thresholds of the time delay characterization script need to be tweaked in order to accurately detect the start of the change in signal.
This can be done by adjusting the stableThreshold and the changingThresholds.
The easiest way to determine what threshold needs to be adjusted is by running the script.
While the script is running, look at the 3 stacked graphs in the bottom-left window of the graph window. These graphs plot the absolute signal change value with indation lines for the stable threshold and the change threshold.
If the change value stays above the stable treshold, even though the system has stabalized, adjust the stable threshold to a value just above the change value line.
If the change value passes the change threshold when it should not, change the change threshold to a higher value.
If the change detection works, but the change time (middle graphs) is set too early, before the signal visibly starts changing, then adjust the stable threshold to a higher value.
The stable threshold shouls always be smaller than the changing threshold.

## Using the calibration
When the script has finished the last measurement it will save a calibration file inside the folder that the script is located in.
This file can be loaded into the Time Delay Calibration (correction) script to correct any dataset that is collected with the gas system.


## Simulation
In line 12, the impulsePy module is imported. If you wish to run this script in a simulation mode, put a hashtag before this line and remove the hashtag before the next line (line 13) which will import the simulator that is included in this repository.


## Known issues
- Only 3 color schemes are set for the different NR pressure measurements, if measurements are done for more than 3 pressures, the list of color schemes should be made longer
- The change graphs (middle) get squished more and more due to the vertical line in every graph getting longer
- Assertion error randomly occurs, until this one is fixed, restart the kernel and run the script again