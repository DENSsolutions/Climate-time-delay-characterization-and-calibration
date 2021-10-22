import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

realData = pd.read_csv('ManualTimeDelayData.csv')

axXpar = 'Pressure'
axXunit = 'mbar'
axYpar = 'Flow'
axYunit = 'mln/min'
axZpar = 'ItP'
axZunit = 's'

raw_data = []
for idnex, row in realData.iterrows():
    dataPoint = [row[axXpar],row[axYpar],row[axZpar]]
    raw_data=raw_data + [dataPoint]

def function(data,a,b,c,d,e):
    x=data[0]
    y=data[1]
    return (a+b*y+c*y**2+d*y**3)+(x*e)/(y**0.5)

x_data = []
y_data = []
z_data = []
for item in raw_data:
    x_data.append(item[0])
    y_data.append(item[1])
    z_data.append(item[2])

# get fit parameters from scipy curve fit
parameters, covariance = curve_fit(function, [x_data, y_data], z_data)

# create surface function model
# setup data points for calculating surface model
model_x_data = np.linspace(min(x_data), max(x_data), 30)
model_y_data = np.linspace(min(y_data), max(y_data), 30)
# create coordinate arrays for vectorized evaluations
X, Y = np.meshgrid(model_x_data, model_y_data)
# calculate Z coordinate array
Z = function(np.array([X, Y]), *parameters)
# setup figure object

fig = plt.figure()
# setup 3d object
ax = Axes3D(fig)
# plot surface
ax.plot_surface(X, Y, Z, cmap="viridis_r", alpha=0.2)
ax.plot_wireframe(X,Y,Z, cmap="viridis_r", linewidth=1)

for index in range(len(x_data)):
    x= x_data[index]
    y=y_data[index]
    z=z_data[index]
    zs = function([x,y],parameters[0],parameters[1],parameters[2],parameters[3],parameters[4])#,parameters[5])#,parameters[6],parameters[7])
    ax.plot([x,x],[y,y],[z,zs],color='red')
    ax.plot(x,y,zs,marker="x", color='red')
    
ax.scatter(x_data, y_data, z_data, depthshade=False, alpha = 1, color='darkgreen')
    

ax.set_xlabel(axXpar + " ("+axXunit+")")
ax.set_ylabel(axYpar + " ("+axYunit+")")
ax.set_zlabel(axZpar + " ("+axZunit+")")

plt.show()
print("Function:")
print(f"({axXpar}*{parameters[4]})*(1/({parameters[0]}+{parameters[1]}*{axYpar}+{parameters[2]}*{axYpar}**2+{parameters[3]}*{axYpar}**3))")
print(" ")