## Pj001_Create_True_Data.py
# ------------------------- #
# Description:
# This project generates true data for the ISS.
# Radar observations will be created for this data and the orbit estimated later on.
# ------------------------- #
# Created by: Ashiv Dhondea, RRSG, UCT.
# Date created: 23 June 2016
# Edits: 24 June 2016: write data to file for ulterior processing.
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
import AstroFunctions as AstFn
import DynamicsFunctions as DynFn
import Num_Integ as ni
# --------------------------- #
# Keplerians extracted from TLE file of ISS on 20 June 2016.
# Keplerians were extracted from the TLE file in MATLAB.
a = 6781.36800424;
e = 0.00005370;
i = np.deg2rad(51.64540000);
BigOmega = np.deg2rad(43.38260000);
omega = np.deg2rad(0.08780000);
nu = np.deg2rad(5.88983219);
# Convert Keplerians to Cartesian state vector
xstate = AstFn.fnKepsToCarts(a,e, i, omega,BigOmega,nu);
# ------------------------------ #
# Declare simulation parameters
dt = 1 # stepsize in [s]
# Find period of ISS
T = AstFn.fnKeplerOrbitalPeriod(a);
t = np.arange(0,T,dt,dtype=np.float64) # simulation period
# Note that the ISS has a period of roughly 90 minutes.
# We are simulating here for a period of 2[h].

# Ystate is the state vector over the entire time interval
Ystate = np.zeros((6,len(t)),dtype = np.float64);
Ystate[:,0] = xstate;

for index in range(1,len(t)):
    # Perform RK4 numerical integration on the J2-perturbed dynamics.
    Ystate[:,index] = ni.fnRK4_vector(DynFn.fnKepler_J2,dt,Ystate[:,index-1],t[index-1]);
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

ax.plot(Ystate[0,:], Ystate[1,:], Ystate[2,:],color='maroon',linewidth='2',label='ISS orbit')
ax.legend()

ax.set_xlabel('x [km]')
ax.set_ylabel('y [km]')
ax.set_zlabel('z [km]')

current_time = time.strftime("%d_%m_%Y_%H_%M_%S");
seq ='_';
created_by_script =os.path.basename(sys.argv[0]);
if created_by_script.endswith('.py'):
    created_by_script = created_by_script[:-3]
ext = '.png'; # or
#ext = '.pdf'
savepath = seq.join([created_by_script,current_time+ext]);
plt.savefig(savepath,bbox_inches='tight');

plt.show()
# ------------------------------------------------- #
# Write to file.
print 'Computations done! \n Now writing to file.'
fname = seq.join([created_by_script+'_truedata_',current_time+'.txt']);
f = open(fname, 'w') # Create data file;
for index in range (0,len(t)):
    for jindex in range (0,6):
        f.write(str(Ystate[jindex,index]));
        f.write(' ');
    f.write('\n');
f.close();
print 'Now writing params file.'
fname = seq.join([created_by_script+'_simulationparams_',current_time+'.txt']);
f = open(fname, 'w') # Create data file;

for index in range (0,len(t)):
    f.write(str(t[index]));
    f.write('\n');

f.close();
print 'Data saved to file. :)'
