## Pj017_CDEKFpaper_Code003.py
# 
# ------------------------- #
# Description:
# This project generates radar observations fromm the iss trajectory in the previous code file.
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
import MathsFunctions as Mth
# --------------------------- #
## Read true data from file
dt = 0.01;#[s] 
simul= 120;#[s] Target observed for this duration.
timevec = np.arange(0,simul,dt,dtype=np.float64) # simulation period is limited to the interval in which the target is seen by the radar.

x_state_SRK = np.zeros([6,len(timevec)],dtype=np.float64); # array of state vectors of the time interval of interest.
index = 0;
with open('Pj017_CDEKFpaper_Code002.txt') as fp:
    for line in fp:
        data = line.split();
        # Place the data read in from the file into the array of state vectors.
        x_state_SRK[:,index] = [np.float64(i) for i in data];
        index+=1;
        if index == len(timevec):
            break
    
fp.close();
print 'True data read'
# ------------------------------------------------------------------------------------------- #
## Observation model parameters.
# Measurement noise covariance matrix
R = np.diag([np.square(50e-3),np.square(np.deg2rad(0.03)),np.square(np.deg2rad(0.03))]); # R contains the variance in the range,azimuth and elevation readings.
# Tk is the time interval between two successive observations. similar to CPI.
Tk = 0.1; # [s]
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
true_P0 = np.diag([1e-6,1e-6,1e-6,1e-6,1e-6,1e-6]);
m = x_state_SRK[:,0]; P = true_P0;
Qc = np.diag([3e-6,3e-6,3e-6])  ; # can be tuned as desired.
# Dispersion matrix
L = np.zeros([6,3],dtype=np.float64);
L[3,0] = 1.0; L[4,1]=1.0;L[5,2]=1.0;
# --------------------------------------------------------- #
## Filtering
x_hat_mcrk4,P_hat_mcrk4 = Flt.fnCD_EKF(m,P,y_meas,R,t_y,40,Qc,L,'1');
x_hat_mfrk4,P_hat_mfrk4 = Flt.fnCD_EKF(m,P,y_meas,R,t_y,40,Qc,L,'2');
x_hat_euler,P_hat_euler = Flt.fnCD_EKF(m,P,y_meas,R,t_y,40,Qc,L,'3');
x_hat_srk2,P_hat_srk2 = Flt.fnCD_EKF(m,P,y_meas,R,t_y,40,Qc,L,'4');
x_hat_srk4,P_hat_srk4 = Flt.fnCD_EKF(m,P,y_meas,R,t_y,40,Qc,L,'5');
print 'Filtering done.'

# -------------------------------------------------------------------------------- #
## Calculate RMSE
se_mcrk4 = np.zeros([6,len(t_y)],dtype=np.float64);
se_mfrk4 = np.zeros([6,len(t_y)],dtype=np.float64);
se_euler = np.zeros([6,len(t_y)],dtype=np.float64);
se_srk2 = np.zeros([6,len(t_y)],dtype=np.float64);
se_srk4 = np.zeros([6,len(t_y)],dtype=np.float64);

print 'Calculating RMSE';

diff_mcrk4 = 0;
diff_mfrk4 = 0;
diff_euler = 0;
diff_srk2 = 0;
diff_srk4 = 0;
                                         
for index in range(0,len(t_y)):
    se_mcrk4[:,index] = np.square(x_hat_mcrk4[:,index] - x_state_SRK[:,Tk*index]);
    se_mfrk4[:,index] = np.square(x_hat_mfrk4[:,index] - x_state_SRK[:,Tk*index]);
    se_euler[:,index] = np.square(x_hat_euler[:,index] - x_state_SRK[:,Tk*index]);
    se_srk2[:,index] = np.square(x_hat_srk2[:,index] - x_state_SRK[:,Tk*index]); 
    se_srk4[:,index] = np.square(x_hat_srk4[:,index] - x_state_SRK[:,Tk*index]); 
    
    diff_mcrk4 += np.sum(se_mcrk4[:,index]);
    diff_mfrk4 += np.sum(se_mfrk4[:,index]);
    diff_euler += np.sum(se_euler[:,index]);
    diff_srk2 += np.sum(se_srk2[:,index]);
    diff_srk4 += np.sum(se_srk4[:,index]);

rmse_mcrk4 = np.sqrt(diff_mcrk4)/np.sqrt(len(t_y)*6.0);
print rmse_mcrk4
rmse_mfrk4= np.sqrt(diff_mfrk4)/np.sqrt(len(t_y)*6.0);
print rmse_mfrk4
rmse_euler = np.sqrt(diff_euler)/np.sqrt(len(t_y)*6.0);
print rmse_euler
rmse_srk2 = np.sqrt(diff_srk2)/np.sqrt(len(t_y)*6.0);
print rmse_srk2 
rmse_srk4 = np.sqrt(diff_srk4)/np.sqrt(len(t_y)*6.0);
print rmse_srk4

##for index in range(0,len(t_y)):
##    P,definiteness = Mth.schol(P_hat_srk4[:,:,index]);
##    if definiteness == 1:
##        # positive definite
##        print 'positive definite'
##    elif definiteness == 0:
##        print 'positive semi-definite'
##    else:
##        print 'negative definite'
            
# --------------------------------------------------------------------------- #
## Plot results
fig = plt.figure()
fig.suptitle('True vs estimated x-position')
plt.plot(timevec,x_state_SRK[0,:],'k',label='x true');
plt.plot(t_y [0:len(t_y)],x_hat_mcrk4[0,:],'b',label='mcrk4');
plt.plot(t_y [0:len(t_y)],x_hat_mfrk4[0,:],'r',label='mfrk4');
plt.plot(t_y [0:len(t_y)],x_hat_euler[0,:],'g',label='euler');
plt.plot(t_y [0:len(t_y)],x_hat_srk2[0,:],'m',label='srk2');
plt.plot(t_y [0:len(t_y)],x_hat_srk4[0,:],'y',label='srk4');
plt.xlabel('t [s]')
plt.ylabel('position [km]')
plt.legend(loc='upper left')
ax = plt.gca()
ax.grid(True)
plt.show()

fig = plt.figure()
fig.suptitle('True vs estimated y-position')
plt.plot(timevec,x_state_SRK[1,:],'k',label='y true');
plt.plot(t_y [0:len(t_y)],x_hat_mcrk4[1,:],'b',label='mcrk4');
plt.plot(t_y [0:len(t_y)],x_hat_mfrk4[1,:],'r',label='mfrk4');
plt.plot(t_y [0:len(t_y)],x_hat_euler[1,:],'g',label='euler');
plt.plot(t_y [0:len(t_y)],x_hat_srk2[1,:],'m',label='srk2');
plt.plot(t_y [0:len(t_y)],x_hat_srk4[1,:],'y',label='srk4');
plt.xlabel('t [s]')
plt.ylabel('position [km]')
plt.legend(loc='upper left')
ax = plt.gca()
ax.grid(True)
plt.show()

fig = plt.figure()
fig.suptitle('True vs estimated z-position')
plt.plot(timevec,x_state_SRK[2,:],'k',label='z true');
plt.plot(t_y [0:len(t_y)],x_hat_mcrk4[2,:],'b',label='mcrk4');
plt.plot(t_y [0:len(t_y)],x_hat_mfrk4[2,:],'r',label='mfrk4');
plt.plot(t_y [0:len(t_y)],x_hat_euler[2,:],'g',label='euler');
plt.plot(t_y [0:len(t_y)],x_hat_srk2[2,:],'m',label='srk2');
plt.plot(t_y [0:len(t_y)],x_hat_srk4[2,:],'y',label='srk4');
plt.xlabel('t [s]')
plt.ylabel('position [km]')
plt.legend(loc='upper left')
ax = plt.gca()
ax.grid(True)
plt.show()

fig = plt.figure()
fig.suptitle('True vs estimated x-vel')
plt.plot(timevec,x_state_SRK[3,:],'k',label='x vel true');
plt.plot(t_y [0:len(t_y)],x_hat_mcrk4[3,:],'b',label='mcrk4');
plt.plot(t_y [0:len(t_y)],x_hat_mfrk4[3,:],'r',label='mfrk4');
plt.plot(t_y [0:len(t_y)],x_hat_euler[3,:],'g',label='euler');
plt.plot(t_y [0:len(t_y)],x_hat_srk2[3,:],'m',label='srk2');
plt.plot(t_y [0:len(t_y)],x_hat_srk4[3,:],'y',label='srk4');
plt.xlabel('t [s]')
plt.ylabel('velocity [km/s]')
plt.legend(loc='upper left')
ax = plt.gca()
ax.grid(True)
plt.show()

fig = plt.figure()
fig.suptitle('True vs estimated y-vel')
plt.plot(timevec,x_state_SRK[4,:],'k',label='y vel true');
plt.plot(t_y [0:len(t_y)],x_hat_mcrk4[4,:],'b',label='mcrk4');
plt.plot(t_y [0:len(t_y)],x_hat_mfrk4[4,:],'r',label='mfrk4');
plt.plot(t_y [0:len(t_y)],x_hat_euler[4,:],'g',label='euler');
plt.plot(t_y [0:len(t_y)],x_hat_srk2[4,:],'m',label='srk2');
plt.plot(t_y [0:len(t_y)],x_hat_srk4[4,:],'y',label='srk4');
plt.xlabel('t [s]')
plt.ylabel('velocity [km/s]')
plt.legend(loc='upper left')
ax = plt.gca()
ax.grid(True)
plt.show()

fig = plt.figure()
fig.suptitle('True vs estimated z-vel')
plt.plot(timevec,x_state_SRK[5,:],'k',label='z vel true');
plt.plot(t_y [0:len(t_y)],x_hat_mcrk4[5,:],'b',label='mcrk4');
plt.plot(t_y [0:len(t_y)],x_hat_mfrk4[5,:],'r',label='mfrk4');
plt.plot(t_y [0:len(t_y)],x_hat_euler[5,:],'g',label='euler');
plt.plot(t_y [0:len(t_y)],x_hat_srk2[5,:],'m',label='srk2');
plt.plot(t_y [0:len(t_y)],x_hat_srk4[5,:],'y',label='srk4');
plt.xlabel('t [s]')
plt.ylabel('velocity [km/s]')
plt.legend(loc='upper left')
ax = plt.gca()
ax.grid(True)
plt.show()

