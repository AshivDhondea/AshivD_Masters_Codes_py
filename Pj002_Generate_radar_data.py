## Pj002_Generate_radar_data.py
# ------------------------- #
# Description:
# This project generates radar observations from the true data data generated in Pj001
# ------------------------- #
# Created by: Ashiv Dhondea, RRSG, UCT.
# Date created: 24 June 2016
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
import time
import sys
import os
# Import Ashiv's own stuff
import AstroConstants as AstCnst
#import AstroFunctions as AstFn
#import DynamicsFunctions as DynFn
#import Num_Integ as ni
import ObservationModels as Obs
# --------------------------- #
## Read true data from files
# First, extract simulation parameters for time
timelist = []; 
with open('Pj001_Create_True_Data_simulationparams__24_06_2016_16_28_47.txt') as fp:
    for line in fp:
        timelist.append(line);
fp.close()
timevec = np.array(timelist,dtype=np.float64); # simulation period.
# Second, extract true data
xtrue = np.zeros([6,np.shape(timevec)[0]],dtype=np.float64); # array of state vectors of the time interval of interest.
index = 0;
with open('Pj001_Create_True_Data_truedata__24_06_2016_16_28_47.txt') as fp:
    for line in fp:
        data = line.split();
        # Place the data read in from the file into the array of state vectors.
        xtrue[:,index] = [np.float64(i) for i in data];
        index+=1;
    
fp.close();
# ------------------------------------------------- #
# Plot results
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_aspect("equal")
mpl.rcParams['legend.fontsize'] = 10

ax = fig.gca(projection='3d')
ax.set_axis_bgcolor('lightsteelblue')
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)

x = AstCnst.R_E* np.outer(np.cos(u), np.sin(v))
y = AstCnst.R_E* np.outer(np.sin(u), np.sin(v))
z = AstCnst.R_E* np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_surface(x, y, z, rstride=4, cstride=4, color='mediumseagreen')

ax.plot(xtrue[0,:], xtrue[1,:], xtrue[2,:],color='maroon',linewidth='2',label='ISS orbit')
ax.legend()

ax.set_xlabel('x [km]')
ax.set_ylabel('y [km]')
ax.set_zlabel('z [km]')

##current_time = time.strftime("%d_%m_%Y_%H_%M_%S");
##seq ='_';
##created_by_script =os.path.basename(sys.argv[0]);
##if created_by_script.endswith('.py'):
##    created_by_script = created_by_script[:-3]
###ext = '.png'; # or
##ext = '.pdf'
##savepath = seq.join([created_by_script,current_time+ext]);
##plt.savefig(savepath,bbox_inches='tight');

plt.show()
# ------------------------------------------------------------------------ #
## Create radar observations.
nObs = 100; # Assumed number of radar measurement vectors.
Xradar = np.zeros([3,nObs],dtype=np.float64);
# Measurement noise covariance matrix
R = np.diag([np.square(0.039),np.square(np.deg2rad(0.03)),np.square(np.deg2rad(0.03))]);

for index in range (0,nObs):
    # Radar observations consist of (range,elevation,azimuth) measurements corrupted by Gaussian noise with covariance matrix R.
    Xradar[:,index] = Obs.fnH(xtrue[0:3,index]) + np.random.multivariate_normal([0,0,0],R);
# ----------------------------------------------------------------------------- #
# Write to file.
print 'Computations done! \n Now writing to file.'
current_time = time.strftime("%d_%m_%Y_%H_%M_%S");
seq ='_';
created_by_script =os.path.basename(sys.argv[0]);
if created_by_script.endswith('.py'):
    created_by_script = created_by_script[:-3]
fname = seq.join([created_by_script+'_radardata_',current_time+'.txt']);
f = open(fname, 'w') # Create data file;
for index in range (0,nObs):
    for jindex in range (0,3):
        f.write(str(Xradar[jindex,index]));
        f.write(' ');
    f.write('\n');
f.close();
