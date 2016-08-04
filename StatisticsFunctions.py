## StatisticsFunctions.py
# ------------------------- #
# Description:
# A collection of functions which implement special maths functions in my Masters dissertation.
# ------------------------- #
# Created by: Ashiv Dhondea, RRSG, UCT.
# Date created: 4 August 2016
# Edits: 
#        
# ------------------------- #
# Import libraries
import numpy as np
from numpy import linalg
import scipy.linalg
# ---------------------------------------------------------------------------------------------- #
## Function to calculate Squared Error
def fnCalc_SE_posvec(x,y):
    # Squared Error
    # Created : 04 August 2016
    
    se_pos = np.zeros([3,np.shape(y)[1]],dtype=np.float64); # in position
    se_vel = np.zeros([3,np.shape(y)[1]],dtype=np.float64); # in velocity

    for index in range(0,np.shape(y)[1]):
        se_pos[:,index] = np.square(x[0:3,index] - y[0:3,index]);
        se_vel[:,index] = np.square(x[3:6,index] - y[3:6,index]);
        
    return se_pos, se_vel

def fnCalc_Total_SE_posvec(se_pos,se_vel):
    # Calculates the total Squared Errors in position and velocity at all instants in the observation window
    # i.e se_pos = 3 x len(t_y) then total_se_pos = len(t_y) x 1 
    # 04 August 2016
    total_se_pos = np.cumsum(se_pos,axis=0)[np.shape(se_pos)[0]-1,:];
    total_se_vel = np.cumsum(se_vel,axis=0)[np.shape(se_vel)[0]-1,:];
    return total_se_pos,total_se_vel

def fnCalc_RMSE(total_se_pos,total_se_vel):
    # Calculates the cumulative squared errors in position and velocity over a time period.
    # i.e. total_se_pos = len(t_y) x 1 then cumulative_se_pos = 1x1
    # 04 August 2016
    cumulative_se_pos = np.cumsum(total_se_pos,axis=0)[np.shape(total_se_pos)[0]-1];
    cumulative_se_vel = np.cumsum(total_se_vel,axis=0)[np.shape(total_se_vel)[0]-1];
    rmse_pos = np.sqrt(cumulative_se_pos);
    rmse_vel = np.sqrt(cumulative_se_vel);
    return rmse_pos,rmse_vel

