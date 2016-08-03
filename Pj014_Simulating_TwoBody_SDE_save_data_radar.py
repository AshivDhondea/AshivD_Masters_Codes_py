## Pj014_Simulating_TwoBody_save_data_radar.py
# Based on Pj002_Generate_radar_data_0.py
# ------------------------- #
# Description:
# This project generates radar observations from the true data data generated in Pj010_Simulating_TwoBody_SDE_save_data.py
# ------------------------- #
# Created by: Ashiv Dhondea, RRSG, UCT.
# Date created: 27 July 2016
# Edits: 
# ------------------------- #
# Theoretical background
# 1. Tracking Filter Engineering, Norman Morrison. 2013.
# 2. Statistical Orbit Determination. Tapley, Schutz, Born. 2004
# 3. Fundamentals of astrodynamics and applications. Vallado. 2001
# ------------------------- #
# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as colors
# Import Ashiv's own stuff
import AstroConstants as AstCnst
import ObservationModels as Obs
# --------------------------- #
## Read true data from file
dt = 0.0001;#[s] changed 29.7.16
t_end = 4;#[s] Target observed only for 2 seconds. changed 29.7.16
timevec = np.arange(0,t_end,dt,dtype=np.float64) # simulation period is limited to the interval in which the target is seen by the radar.

xtrue = np.zeros([6,len(timevec)],dtype=np.float64); # array of state vectors of the time interval of interest.
index = 0;
with open('Pj010_Simulating_TwoBody_SDE_save_data.txt') as fp:
    for line in fp:
        data = line.split();
        # Place the data read in from the file into the array of state vectors.
        xtrue[:,index] = [np.float64(i) for i in data];
        index+=1;
        if index == len(timevec):
            break
    
fp.close();
# ------------------------------------------------- #
## Observation model parameters.
del_y =100;#decimation in time
t_y = np.arange(0,t_end,dt*del_y,dtype=np.float64); # time vector for measurements.
y_meas = np.zeros([3,len(t_y)],dtype=np.float64); # measurement vector.

# Measurement noise covariance matrix
R = np.diag([np.square(50e-3),np.square(np.deg2rad(0.1)),np.square(np.deg2rad(0.1))]); # R contains the variance in the range,azimuth and elevation readings.
# ------------------------------------------------------------------------ #
## Create radar observations.
xradar = np.zeros([6,len(t_y)],dtype=np.float64); # This contains the true radar data for the period of observation.

for index in range(0,len(t_y)): 
    # sensor measurements are contaminated with awgn of covariane R.
    vn = np.random.multivariate_normal([0,0,0],R);
    pos = np.array([xtrue[0,index*(del_y)],xtrue[1,index*(del_y)],xtrue[2,index*(del_y)]],dtype=np.float64);
    y_meas[:,index] = Obs.fnH(pos)+ vn;
    xradar[:,index] = xtrue[:,index*(del_y)];
# ----------------------------------------------------------------------------- #
# Write to file.
print 'Computations done! \n Now writing radar data to file.'

fname = 'Pj014_Simulating_TwoBody_save_data_radar.txt';
f = open(fname, 'w') # Create data file;
for index in range (0,len(t_y)):
    for jindex in range (0,3):
        f.write(str(y_meas[jindex,index]));
        f.write(' ');
    f.write('\n');
f.close();

print 'Computations done! \n Now writing true data to file.'

fname = 'Pj014_Simulating_TwoBody_save_data_radar_true.txt';
f = open(fname, 'w') # Create data file;
for index in range (0,len(t_y)):
    for jindex in range (0,6):
        f.write(str(xradar[jindex,index]));
        f.write(' ');
    f.write('\n');
f.close();
