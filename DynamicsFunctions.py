# ------------------------- #
# Description:
# A collection of functions which implement dynamics-related functions for orbiting objects.
# ------------------------- #
# Created by: Ashiv Dhondea, RRSG, UCT.
# Date created: 20 June 2016
# Edits:
# 21/06/2016 : cleaned up array indices in fnKepler_J2_augmented.
# 23/06/16 : fixed function fnKepler_J2_augmented. Validated against its MATLAB counterpart.
# 23/06/16: added in 3 functions for polynomial models. (i.e. kinematic models).
# 7 July 2016: added function to generate nominal trajectory for Two-Body problem.
# 14 July 2016: added function to implement differential equations for the mean and covariance matrix. Integrated by RK4 for the MC-RK4 method for CD-EKF.
# 15 July 2016: edited yesterday's function.
# 16 July 2016: edited the function fnMoment_DE to be integrated by fnRK4_moment(f, dt, x,t,L,Q)
# 19 July 2016: edited the function fn_Create_Concatenated_Block_Diag_Matrix. Probably a better implementation.
# ------------------------- #
# Import libraries
import numpy as np
from numpy import linalg
import scipy
# Ashiv's own libraries
import AstroConstants as AstCnst
import Num_Integ as ni
# -------------------------------------------------------------------- #
def fnKepler_J2(t,X):
    # emulates fnKepler_J2.m. 20 June 2016
    r = np.linalg.norm(X[0:3]);
    Xdot = np.zeros([6],dtype=np.float64);
    Xdot[0:3] = X[3:6]; # dx/dt = xdot; dy/dt = ydot; dz/dt = zdot
    expr = 1.5*AstCnst.J_2*(AstCnst.R_E/r)**2 ;

    Xdot[3] = -(AstCnst.mu_E/r**3)*X[0]*(1-expr*(5*(X[2]/r)**2 -1));
    Xdot[4] = -(AstCnst.mu_E/r**3)*X[1]*(1-expr*(5*(X[2]/r)**2 -1));
    Xdot[5] = -(AstCnst.mu_E/r**3)*X[2]*(1-expr*(5*(X[2]/r)**2 -3));
    return Xdot

def fnKepler_J2_augmented(t,X):
    # validated against the equivalent MATLAB function on 23 June 2016.
    r = np.linalg.norm(X[0:3]);
    # state vector augmented by reshaped state transition matrix
    Xdot = np.zeros([6 + 6*6],dtype=np.float64);
    Xdot[0:3] = X[3:6]; # dx/dt = xdot; dy/dt = ydot; dz/dt = zdot
    expr = 1.5*AstCnst.J_2*(AstCnst.R_E/r)**2 ;
    expr1 = -(AstCnst.mu_E/r**3)*(1 - expr*(5*((X[2]/r)**2) -1));
    
    Xdot[3] = expr1*X[0];
    Xdot[4] = expr1*X[1];   
    Xdot[5] = -(AstCnst.mu_E/r**3)*X[2]*(1-expr*(5*(X[2]/r)**2 -3));
    # The state transition matrix's elements are the last 36 elements in the input state vector
    Phi = np.reshape(X[6:6+6*6+1],(6,6)); # extract Phi matrix from the input vector X
    
    # Find matrix A (state sensitivity matrix)
    # We start with a matrix of zeros and then add in the non-zero elements.
    Amatrix = np.zeros([6,6],dtype=np.float64);
    Amatrix[0,3]=1.0;
    Amatrix[1,4]=1.0;
    Amatrix[2,5]=1.0;

    expr2 = 3*(AstCnst.mu_E/r**5)*(1 - 2.5*AstCnst.J_2*(AstCnst.R_E/r)**2*(7*(X[2]/r)**2) -1); # fixed
    Amatrix[3,0] = expr1 + expr2*X[0]**2; 
    Amatrix[3,1] = expr2*X[0]*X[1]; 
    expr3 =  3*(AstCnst.mu_E/r**5)*(1 - 2.5*AstCnst.J_2*(AstCnst.R_E/r)**2*(7*(X[2]/r)**2) -3);
    Amatrix[3,2] = expr3*X[0]*X[2];
    
    Amatrix[4,0] = Amatrix[3,1];
    Amatrix[4,1] = expr1 + expr2*X[1]**2;
    Amatrix[4,2] = expr3*X[1]*X[2];

    Amatrix[5,0] = Amatrix[3,2];
    Amatrix[5,1] = Amatrix[4,2];
    Amatrix[5,2] =-(AstCnst.mu_E/r**3)*(1-expr*(5*(X[2]/r)**2 -3)) + 3*(AstCnst.mu_E/r**5)*(1 - 2.5*AstCnst.J_2*(AstCnst.R_E/r)**2*(7*(X[2]/r)**2) -5)*X[2]**2;
    
    # The state transition matrix's differential equation.
    PhiDot = np.dot(Amatrix,Phi); # Amatrix is a time dependent variable.i.e. not constant
    Xdot[6:6+6*6+1] = np.reshape(PhiDot,6*6);
    return Xdot

def fnGenerate_Nominal_Trajectory(Xdash,timevec):
    dt = timevec[1]-timevec[0];
    Xstate = np.zeros([6+6*6,len(timevec)],dtype=np.float64);
    Xstate[0:6,0] = Xdash;
    Xstate[6:,0] = np.reshape(np.eye(6,dtype=np.float64),6*6);
    
    for index in range(1,len(timevec)):
        # Perform RK4 numerical integration on the J2-perturbed dynamics.
        Xstate[:,index] = ni.fnRK4_vector(fnKepler_J2_augmented,dt,Xstate[:,index-1],timevec[index-1]);
    return Xstate
# --------------------------------------------------------------------------------------------#
## Polynomial model functions
def fn_Generate_STM_polynom(zeta,nStates):
    # fn_Generate_STM_polynom creates the state transition matrix for polynomial models 
    # of degree (nStates-1) over a span of transition of zeta [s].
    # Polynomial models are a subset of the class of constant-coefficient linear DEs.
    # Refer to: Tracking Filter Engineering, Norman Morrison.
    stm = np.eye(nStates,dtype=float);
    for yindex in range (0,nStates):
        for xindex in range (yindex,nStates): # STM is upper triangular
            stm[yindex,xindex] = np.power(zeta,xindex-yindex)/float(math.factorial(xindex-yindex));
    return stm;     

def fn_Generate_STM_polynom_3D(zeta,nStates,dimensionality):
    # fn_Generate_STM_polynom_3D generates the full state transition matrix for 
    # the required dimensionality.
    stm = fn_Generate_STM_polynom(dt,nStates);
    stm3 = fn_Create_Concatenated_Block_Diag_Matrix(stm,dimensionality-1);
    return stm3;

def fn_Create_Concatenated_Block_Diag_Matrix(R,stacklen):
    # fn_Create_Concatenated_Block_Diag_Matrix creates a block diagonal matrix of size (stacklen) x (stacklen)
    # whose diagonal blocks are copies of the matrix R.
    ##L = [R]; 
    ##for index in range (0,stacklen):
    ##    L.append(R);
    ##    ryn = scipy.linalg.block_diag(*L);
    ryn =np.kron(np.eye(stacklen),R); # Edit: 19/07/2016: probably better idea than using a for loop.
    return ryn
# -------------------------------------------------------------------------------------------------------------- #
## fnMoment_DE implements the DEs for propagation of the mean and covariance.
    """ Based on "Various ways to compute the continuous-discrete extended Kalman filter"
     @article{frogerais2012various,
      title={Various ways to compute the continuous-discrete extended Kalman filter},
      author={Frogerais, Paul and Bellanger, Jean-Jacques and Senhadji, Lotfi},
      journal={Automatic Control, IEEE Transactions on},
      volume={57},
      number={4},
      pages={1000--1004},
      year={2012},
      publisher={IEEE}
    } """
def fnMoment_DE(t,X):
    # 14 July 2016: moment propagation differential equations. Based on Frogerais 2012.
    # Function called by CD-EKF MC-RK4 (fnCD_EKF_predict_MC_RK4)
    # Code for computing mean state vector and Jacobian matrix are obtained from the function fnKepler_J2_augmented

    # t is the time.
    # X is the mean state vector augmented by the reshaped covariance matrix.
    # L is the dispersion matrix.
    # Q is the process noise covariance matrix.
    L = np.zeros([6,3],dtype=np.float64);
    L[3,0] = 1.0; L[4,1]=1.0;L[5,2]=1.0;
    Q = np.diag([2.4064e-6,2.4064e-6,2.4064e-6]);
    
    r = np.linalg.norm(X[0:3]);  

    # state vector augmented by reshaped state transition matrix
    Xdot = np.zeros([6 + 6*6],dtype=np.float64);
    Xdot[0:3] = X[3:6]; # dx/dt = xdot; dy/dt = ydot; dz/dt = zdot
    expr = 1.5*AstCnst.J_2*(AstCnst.R_E/r)**2 ;
    expr1 = -(AstCnst.mu_E/r**3)*(1 - expr*(5*((X[2]/r)**2) -1));
    
    Xdot[3] = expr1*X[0];
    Xdot[4] = expr1*X[1];   
    Xdot[5] = -(AstCnst.mu_E/r**3)*X[2]*(1-expr*(5*(X[2]/r)**2 -3));

    # Extract covariance matrix from state vector.
    P = np.reshape(X[6:6+6*6+1],(6,6));
    # Find matrix A (state sensitivity matrix)
    # We start with a matrix of zeros and then add in the non-zero elements.
    Amatrix = np.zeros([6,6],dtype=np.float64);
    Amatrix[0,3]=1.0;
    Amatrix[1,4]=1.0;
    Amatrix[2,5]=1.0;

    expr2 = 3*(AstCnst.mu_E/r**5)*(1 - 2.5*AstCnst.J_2*(AstCnst.R_E/r)**2*(7*(X[2]/r)**2) -1); # fixed
    Amatrix[3,0] = expr1 + expr2*X[0]**2; 
    Amatrix[3,1] = expr2*X[0]*X[1]; 
    expr3 =  3*(AstCnst.mu_E/r**5)*(1 - 2.5*AstCnst.J_2*(AstCnst.R_E/r)**2*(7*(X[2]/r)**2) -3);
    Amatrix[3,2] = expr3*X[0]*X[2];
    
    Amatrix[4,0] = Amatrix[3,1];
    Amatrix[4,1] = expr1 + expr2*X[1]**2;
    Amatrix[4,2] = expr3*X[1]*X[2];

    Amatrix[5,0] = Amatrix[3,2];
    Amatrix[5,1] = Amatrix[4,2];
    Amatrix[5,2] =-(AstCnst.mu_E/r**3)*(1-expr*(5*(X[2]/r)**2 -3)) + 3*(AstCnst.mu_E/r**5)*(1 - 2.5*AstCnst.J_2*(AstCnst.R_E/r)**2*(7*(X[2]/r)**2) -5)*X[2]**2;

    P = np.dot(Amatrix,P) + np.transpose(np.dot(P,Amatrix)) + np.dot(np.dot(L,Q),np.transpose(L));
    # to force symmetry for covariance matrix P.
    P = 0.5*(P + np.transpose(P));
    
    Xdot[6:6+6*6+1] = np.reshape(P,6*6);
    return Xdot;

def fnJacobian_Kepler_J2(X):
    r = np.linalg.norm(X[0:3]);
    expr = 1.5*AstCnst.J_2*(AstCnst.R_E/r)**2 ;
    expr1 = -(AstCnst.mu_E/r**3)*(1 - expr*(5*((X[2]/r)**2) -1));
    # Find matrix A (state sensitivity matrix)
    # We start with a matrix of zeros and then add in the non-zero elements.
    Amatrix = np.zeros([6,6],dtype=np.float64);
    Amatrix[0,3]=1.0;
    Amatrix[1,4]=1.0;
    Amatrix[2,5]=1.0;

    expr2 = 3*(AstCnst.mu_E/r**5)*(1 - 2.5*AstCnst.J_2*(AstCnst.R_E/r)**2*(7*(X[2]/r)**2) -1); # fixed
    Amatrix[3,0] = expr1 + expr2*X[0]**2; 
    Amatrix[3,1] = expr2*X[0]*X[1]; 
    expr3 =  3*(AstCnst.mu_E/r**5)*(1 - 2.5*AstCnst.J_2*(AstCnst.R_E/r)**2*(7*(X[2]/r)**2) -3);
    Amatrix[3,2] = expr3*X[0]*X[2];
    
    Amatrix[4,0] = Amatrix[3,1];
    Amatrix[4,1] = expr1 + expr2*X[1]**2;
    Amatrix[4,2] = expr3*X[1]*X[2];

    Amatrix[5,0] = Amatrix[3,2];
    Amatrix[5,1] = Amatrix[4,2];
    Amatrix[5,2] =-(AstCnst.mu_E/r**3)*(1-expr*(5*(X[2]/r)**2 -3)) + 3*(AstCnst.mu_E/r**5)*(1 - 2.5*AstCnst.J_2*(AstCnst.R_E/r)**2*(7*(X[2]/r)**2) -5)*X[2]**2;
    return Amatrix


    


    
