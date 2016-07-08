## This project breaks down the code in the file below into functions. 07/07/16
# Building upon Pj009.
# Now, run the Gauss-Newton in 
## Pj003_Scenario0_GaussNewtonFilter.py
# ------------------------- #
# Description:
# This project generates radar observations from the true data data generated in Pj001
# ------------------------- #
# Created by: Ashiv Dhondea, RRSG, UCT.
# Date created: 24 June 2016
# Edits: 7 July 2016: fixed a bug related to Phi.
# ------------------------- #
# Theoretical background
# 1. Tracking Filter Engineering, Norman Morrison. 2013.
# 2. Statistical Orbit Determination. Tapley, Schutz, Born. 2004
# 3. Fundamentals of astrodynamics and applications. Vallado. 2001
# 4. Crassidis, Junkins. Optimal estimation of dynamics systems.2011
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
import ObservationModels as Obs
import Filters as Flt
# -------------------------------------------------------------------------- #
## Explanation
# We will consider a few different scenarios of increasing levels of complexity.
# Scenario 0: Radar is placed at the centre of an Earth which is transparent and does not rotate.
# Scenario 1: Radar is placed at a location on the surface of the Earth.
# Scenario 0 is considered in this project file.
# To further simplify the task here, the track initialization problem is nullified.
# We assume availability of a suitable initial track. Later on, we will worry about track initialization and initial orbit determination.
# --------------------------------------------------------------------------- #
# 29 June 2016
# We now develop the Gauss-Newton filter in Case 4 to operate in an iterative way.
# The iteration is done to refine the state estimates obtained in a single run of the filter.
# ------------------------------------------------------------------------------------------- #
## Read true data from files
# First, extract simulation parameters for time
timelist = []; 
with open('Pj001_Create_True_Data_simulationparams__24_06_2016_16_28_47.txt') as fp:
    for line in fp:
        timelist.append(line);
fp.close()
timevec = np.array(timelist,dtype=np.float64); # simulation period.

# Second, extract true data
nObs = 100; # Assumed number of radar measurement vectors.
Xradar = np.zeros([3,nObs],dtype=np.float64);
index = 0;
with open('Pj002_Generate_radar_data_radardata__24_06_2016_21_45_15.txt') as fp:
    for line in fp:
        data = line.split();
        # Place the data read in from the file into the array of state vectors.
        Xradar[:,index] = [np.float64(i) for i in data];
        index+=1;
    
fp.close();
# ---------------------------------------------------------------------------------------------- #
# Maximum iteration depth (to prevent infinite looping)
iter_max = 1;
L = 20; # Filter window length (can be changed)

xnom_ini = np.array([4600.53917168,4950.79940098,553.772365323,-3.83039035655,2.89058678525,5.9797711844],dtype=np.float64);
Xdash = xnom_ini;

# ------------------------------------------------------------------------------- #
M = np.zeros([3,6],dtype=np.float64);
# Measurement noise covariance matrix
R = np.diag([np.square(0.039),np.square(np.deg2rad(0.03)),np.square(np.deg2rad(0.03))]);
Rinv = np.linalg.inv(R);
RynInv = DynFn.fn_Create_Concatenated_Block_Diag_Matrix(Rinv,L-1);
Ryn = DynFn.fn_Create_Concatenated_Block_Diag_Matrix(R,L-1);

epsilon = 1;

for iter_num in range (0,iter_max):
    Xdash, S_hat,Ji = Flt.fnGaussNewtonBatch(Xdash,timevec,L,M,Xradar,RynInv,Ryn,DynFn.fnGenerate_Nominal_Trajectory,Obs.fnH,Obs.fnJacobianH,Flt.fnMVA)
    print Xdash
    # Retrodict Xdash back to initial time step.
    timeretro = np.flipud(timevec[0:L]);
    Xout = DynFn.fnGenerate_Nominal_Trajectory(Xdash,timeretro);
    Xdash = Xout[0:6,L-1];
    
    if iter_num == 0:
        Ji_prior = Ji;
        print 'First iteration'
        continue
    else:
        if Ji - Ji_prior > 0:
            # Cost has increased, so failed to converge
            print 'Experienced convergence difficulties.'
            break
        else: # Cost has decreased or stayed the same.
            # Stopping rule
            delta_J = abs(Ji - Ji_prior)/Ji; # Update change in cost
            if delta_J < epsilon/np.linalg.norm(RynInv):
                X_hat = Xdash; # Filter has converged, so the answer Xdash is good enough to be used as X_hat
                print 'Converged'
                break
            else:
                print 'Iterating';
                print iter_num;
                # Update cost for next iteration.
                Ji_prior = Ji;
                continue
    

print 'done'
