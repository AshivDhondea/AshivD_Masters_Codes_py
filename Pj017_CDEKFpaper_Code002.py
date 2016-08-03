## Pj017_CDEKFpaper_Code002.py
# 
# ------------------------- #
# Description:
# This project simulates the trajectory of the ISS by numerically integrating the J2-perturbed two-body SDE.
# ISS TLE downloaded from celestrak.com on 27/07/16. Then radar observations of the 
# ------------------------- #
# Created by: Ashiv Dhondea, RRSG, UCT.
# Date created: 31 July 2016
# Edits: 
# ------------------------- #
# Theoretical background
# 1. Basic tracking using nonlinear continuous-time dynamic models [Tutorial], Crouse(2015)
# 2. Lecture notes on applied stochastic differential equations. Sarkka,Solin (2014).
# 3. ORBIT DETERMINATION OF LEO SATELLITES FOR A SINGLE PASS THROUGH A RADAR: COMPARISON OF METHODS Z. Khutorovsky, S. Kamensky and, N. Sbytov

"""
@article{crouse2015basic,
  title={Basic tracking using nonlinear continuous-time dynamic models [Tutorial]},
  author={Crouse, David},
  journal={Aerospace and Electronic Systems Magazine, IEEE},
  volume={30},
  number={2},
  pages={4--41},
  year={2015},
  publisher={IEEE}
}
"""
# ------------------------- #
# Import libraries
import numpy as np
import matplotlib.pyplot as plt

# Import Ashiv's own stuff
import AstroConstants as AstCnst
import AstroFunctions as AstFn
import DynamicsFunctions as DynFn
import ObservationModels as Obs
import Num_Integ as Integ
import Filters as Flt
# --------------------------- #
# Keplerians extracted from TLE file of ISS on 27 July 2016.
# Keplerians were extracted from the TLE file in MATLAB.
a = 6780.61843448;
e = 0.00022100;
i = np.deg2rad(51.64320000);
BigOmega = np.deg2rad(217.98080000);
omega = np.deg2rad(82.55250000);
nu = np.deg2rad(2.15845710);
# Convert Keplerians to Cartesian state vector
xstate = AstFn.fnKepsToCarts(a,e, i, omega,BigOmega,nu);
# ------------------------------------------------- #
## Declare variables
# Dispersion matrix
L = np.zeros([6,3],dtype=np.float64);
L[3,0] = 1.0; L[4,1]=1.0;L[5,2]=1.0;

t = 0;
true_m0 = xstate;
true_P0 = np.diag([1e-6,1e-6,1e-6,1e-6,1e-6,1e-6]); 

x = true_m0 + np.random.multivariate_normal(np.zeros(np.shape(true_m0)[0],dtype=np.float64),true_P0);

dt = 0.01;#[s]
simul = 120; 
timevec = np.arange(t,simul,dt,dtype=np.float64) # simulation period
Qc = np.diag([3e-6,3e-6,3e-6])  ; # can be tuned as desired.

Qd = dt*Qc;
# ------------------------------------------------------------------------ #
## The Order 1.5 Strong Method for Additive Noise 
x_state_SRK = Integ.fnSRK_Crouse(x,DynFn.fnKepler_J2,timevec,L,Qd);
print 'Order 1.5 strong SRK done.'
#x_state_SRK = Integ.fnEuler_Maruyama(x,DynFn.fnKepler_J2,timevec,L,Qd);

# ------------------------------------------------------------------------ #
## Plot results
fig = plt.figure()
fig.suptitle('True position')
plt.plot(timevec,x_state_SRK[0,:],'b',label='x pos true');
plt.plot(timevec,x_state_SRK[1,:],'r',label='y pos true');
plt.plot(timevec,x_state_SRK[2,:],'g',label='z pos true');
plt.xlabel('t [s]')
plt.ylabel('position [km]')
plt.legend(loc='upper left')
ax = plt.gca()
ax.grid(True)
plt.show()

fig = plt.figure()
fig.suptitle('True velocity')
plt.plot(timevec,x_state_SRK[3,:],'b',label='x vel true');
plt.plot(timevec,x_state_SRK[4,:],'r',label='y vel true');
plt.plot(timevec,x_state_SRK[5,:],'g',label='z vel true');
plt.xlabel('t [s]')
plt.ylabel('velocity [km/s]')
plt.legend(loc='upper left')
ax = plt.gca()
ax.grid(True)
plt.show()
# ------------------------------------------------------------------------ #
## Real data simulated.
# We can now create radar observations in the next project file. We write the data to file for now.
# ------------------------------------------------------------------------ #
fname = 'Pj017_CDEKFpaper_Code002.txt';
f = open(fname, 'w') # Create data file;
for index in range (0,len(timevec)):
    for jindex in range (0,6):
        f.write(str(x_state_SRK[jindex,index]));
        f.write(' ');
    f.write('\n');
f.close();

print 'Data saved to file.'
