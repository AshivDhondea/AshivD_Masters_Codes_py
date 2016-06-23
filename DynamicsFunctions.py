## DynamicsFunctions.py
# ------------------------- #
# Description:
# A collection of functions which implement dynamics function for orbiting objects.
# ------------------------- #
# Created by: Ashiv Dhondea, RRSG, UCT.
# Date created: 20 June 2016
# Edits:
# 21/06/2016 : cleaned up array indices in fnKepler_J2_augmented.
# 23/06/16 : fixed function fnKepler_J2_augmented. Validated against its MATLAB counterpart.
# ------------------------- #
# Import libraries
import numpy as np
import AstroConstants as AstCnst
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

