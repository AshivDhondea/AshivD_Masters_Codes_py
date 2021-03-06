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
L = 5; # Filter window length (can be changed)

xnom_ini = np.array([4600.53917168,4950.79940098,553.772365323,-3.83039035655,2.89058678525,5.9797711844],dtype=np.float64);
Xdash = xnom_ini;


def fnGenerate_Nominal_Trajectory(Xdash,timevec):
    dt = timevec[1]-timevec[0];
    Xstate = np.zeros([6+6*6,len(timevec)],dtype=np.float64);
    Xstate[0:6,0] = Xdash;
    Xstate[6:,0] = np.reshape(np.eye(6,dtype=np.float64),6*6);
    
    for index in range(1,len(timevec)):
        # Perform RK4 numerical integration on the J2-perturbed dynamics.
        Xstate[:,index] = ni.fnRK4_vector(DynFn.fnKepler_J2_augmented,dt,Xstate[:,index-1],timevec[index-1]);
    return Xstate
# -------------------------------------------------------------------------- #
def fnMVA( Lambda, TotalObservationMatrixTranspose, RynInv, Ryn ,delta_Y ):
    # FNMVA implements the Minimum Variance Algorithm (MVA).
    # Estimated covariance matrix
    #S_hat = np.linalg.inv(Lambda);
    # Filter matrix W in TFE == matrix M in Tapley,Schutz,Born.
    #W = np.dot(np.dot(S_hat,TotalObservationMatrixTranspose),RynInv);
    A = Lambda;
    b = np.dot(TotalObservationMatrixTranspose,RynInv);
    W = np.linalg.solve(A,b);
    S_hat = np.dot(np.dot(W,Ryn),np.transpose(W));
    # Estimated perturbation vector
    delta_X_hat = np.dot(W,delta_Y);
    return delta_X_hat,S_hat
# ------------------------------------------------------------------------------- #
M = np.zeros([3,6],dtype=np.float64);
# Measurement noise covariance matrix
R = np.diag([np.square(0.039),np.square(np.deg2rad(0.03)),np.square(np.deg2rad(0.03))]);
Rinv = np.linalg.inv(R);
RynInv = DynFn.fn_Create_Concatenated_Block_Diag_Matrix(Rinv,L-1);
Ryn = DynFn.fn_Create_Concatenated_Block_Diag_Matrix(R,L-1);

for iter_num in range (0,iter_max):
    # From initial estimate of state vector xdash, generate a nominal trajectory over L (the filter window length)
    Xnom = fnGenerate_Nominal_Trajectory(Xdash,timevec[0:L]);
    # Build total observation matrix
    # Total Observation Matrix
    TotalObservationMatrix = np.zeros([3*L,6],dtype=np.float64);
    TotalObservationMatrixTranspose = np.zeros([6,3*L],dtype=np.float64); 

    # Initialize the matrices
    M[0:3,0:3] = Obs.fnJacobianH( Xnom[0:3,0] );
    Phi = np.reshape(Xnom[6:,0],(6,6));
    TotalObservationMatrix[0:3,:] = np.dot(M,Phi); 
    TotalObservationMatrixTranspose[:,0:3] = np.transpose(np.dot(M,Phi));
    
    TotalObservationVector = np.zeros([3*L],dtype=np.float64);
    # Load the sensor input into the TotalObservationVector at the validity instant Tvi == tstart
    TotalObservationVector[0:3] =  Xradar[0:3,0];

    # delta Y vector == simulated perturbation vector in TFE
    delta_Y = np.zeros([3*L],dtype=np.float64);

    # Load the starting sample into the simulated perturbation vector
    delta_Y[0:3] = np.transpose(TotalObservationVector[0:3] - Obs.fnH( Xnom[0:3,0]) ); 

    # Load the filter's input stack
    for index in range(1,L):# tstart+1:tstop
        M[0:3,0:3] = Obs.fnJacobianH( Xnom[0:3,index] );
        Phi = np.reshape(Xnom[6:,index],(6,6));
        # The T matrix and its transpose
        TotalObservationMatrix[3*index:3*index+3,:] = np.dot(M,Phi);
        TotalObservationMatrixTranspose[:,3*index:3*index+3] = np.transpose(np.dot(M,Phi));

        TotalObservationVector[3*index:3*index+3] =  Xradar[0:3,index];
        delta_Y[3*index:3*index+3] = np.transpose(TotalObservationVector[3*index:3*index+3] - Obs.fnH( Xnom[0:3,index]) );

    # Fisher information matrix
    Lambda = np.dot(np.dot(TotalObservationMatrixTranspose,RynInv),TotalObservationMatrix);
    delta_X_hat,S_hat = fnMVA( Lambda, TotalObservationMatrixTranspose, RynInv, Ryn,delta_Y )

    Xdash = Xnom[0:6,L-1] + delta_X_hat;   

print 'done'
