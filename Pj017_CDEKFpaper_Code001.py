## Pj017_CDEKFpaper_Code001.py
# 
# ------------------------- #
# Description:
# This project simulates the trajectory of the ISS by numerically integrating the J2-perturbed two-body SDE.
# ISS TLE downloaded from celestrak.com on 27/07/16. Then radar observations of the 
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
## Real data simulated.
# We can now create radar observations.
# ------------------------------------------------------------------------ #
## Observation model parameters.
# Measurement noise covariance matrix
R = np.diag([np.square(50e-3),np.square(np.deg2rad(0.1)),np.square(np.deg2rad(0.1))]); # R contains the variance in the range,azimuth and elevation readings.
# Tk is the time interval between two successive observations. similar to CPI.
Tk = 8; # [s]
t_y = np.arange(0,simul,Tk,dtype=np.float64); # time vector for measurements.

y_meas = np.zeros([3,len(t_y)],dtype=np.float64); # measurement vector.
# ------------------------------------------------------------------------ #
## Create radar observations.
for index in range(0,len(t_y)): 
    # sensor measurements are contaminated with awgn of covariane R.
    vn = np.random.multivariate_normal([0.0,0.0,0.0],R);
    pos = np.array([x_state_SRK[0,index*(Tk)],x_state_SRK[1,index*(Tk)],x_state_SRK[2,index*(Tk)]],dtype=np.float64);
    y_meas[:,index] = Obs.fnH(pos)+ vn;

print 'Radar observations created.'
# ---------------------------------------------------------------------------- #
# Initial state estimate and initial covariance matrix estimate
# arrays to hold estimates.
x_hat = np.zeros([6,len(t_y)],dtype=np.float64);
P_hat = np.zeros([6,6,len(t_y)],dtype=np.float64);
m = x; P = true_P0;
# --------------------------------------------------------- #
## Filtering
x_hat,P_hat = Flt.fnEKF(m,P,y_meas,R,t_y,10,Qc,L,'1');
x_hat_n,P_hat_n = Flt.fnEKF(m,P,y_meas,R,t_y,20,Qc,L,'2');
x_hat_euler,P_hat_euler = Flt.fnEKF(m,P,y_meas,R,t_y,1,Qc,L,'3');
x_hat_srk4,P_hat_srk4 = Flt.fnEKF(m,P,y_meas,R,t_y,1,Qc,L,'5');
print 'Filtering done.'

## Calculate RMSE
rmse_pos = np.zeros(len(t_y),dtype=np.float64);
rmse_vel = np.zeros(len(t_y),dtype=np.float64);
diff = 0;
diff_n = 0;
diff_euler = 0;
diff_srk4 = 0;

print 'Calculating RMSE';
#print np.dot(np.diag(P_hat[:,:,0]),np.eye(np.shape(x_hat)[0]))
                                          
for index in range(0,len(t_y)):
    diff += np.sum(np.square(x_hat[:,index] - x_state_SRK[:,index]));
    diff_n += np.sum(np.square(x_hat_n[:,index] - x_state_SRK[:,index]));
    diff_euler += np.sum(np.square(x_hat_euler[:,index] - x_state_SRK[:,index]));
    diff_srk4 += np.sum(np.square(x_hat_srk4[:,index] - x_state_SRK[:,index]));
##    rmse_pos[index] = np.linalg.norm(x_hat[0:3,index] - x_state_SRK[0:3,index]) / np.sqrt(len(t_y));
##    rmse_vel[index] = np.linalg.norm(x_hat[3:,index] - x_state_SRK[3:,index]) / np.sqrt(len(t_y));

rmse = np.sqrt(diff)/np.sqrt(len(t_y)*3.0);
print rmse
rmse_n = np.sqrt(diff_n)/np.sqrt(len(t_y)*3.0);
print rmse_n
rmse_euler = np.sqrt(diff_euler)/np.sqrt(len(t_y)*3.0);
print rmse_euler
rmse_srk4 = np.sqrt(diff_srk4)/np.sqrt(len(t_y)*3.0);
print rmse_srk4 
