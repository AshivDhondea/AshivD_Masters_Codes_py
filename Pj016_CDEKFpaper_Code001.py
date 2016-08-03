## Pj016_CDEKFpaper_Code001.py
# Based on  Pj010_Simulating_TwoBody_SDE_save_data.py
# ------------------------- #
# Description:
# This project simulates the trajectory of the ISS by numerically integrating the J2-perturbed two-body SDE.
# ISS TLE downloaded from celestrak.com on 27/07/16.
# ------------------------- #
# Created by: Ashiv Dhondea, RRSG, UCT.
# Date created: 30 July 2016
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
true_P0 = np.diag([1e-4,1e-4,1e-4,1e-4,1e-4,1e-4]);

x = true_m0 + np.random.multivariate_normal(np.zeros(np.shape(true_m0)[0],dtype=np.float64),true_P0);

dt = 0.001;#[s]
simul = 60*5; # simulate for 5 minutes. Following Khutorovsky
timevec = np.arange(t,simul,dt,dtype=np.float64) # simulation period
Qc = np.diag([2.4064e-4,2.4064e-4,2.4064e-4])  ; # can be tuned as desired.

Qd = dt*Qc;
# ------------------------------------------------------------------------ #
## The Order 1.5 Strong Method for Additive Noise 
x_state_SRK = Integ.fnSRK_Crouse(x,DynFn.fnKepler_J2,timevec,L,Qd);
print 'Order 1.5 strong SRK done.'
# ------------------------------------------------------------------------ #
# Write to file.
print 'Computations done! \n Now writing to file.'
fname = 'Pj016_CDEKFpaper_Code001.txt';
f = open(fname, 'w') # Create data file;
for index in range (0,len(timevec)):
    for jindex in range (0,6):
        f.write(str(x_state_SRK[jindex,index]));
        f.write(' ');
    f.write('\n');
f.close();

print 'Data saved to file. :)'
