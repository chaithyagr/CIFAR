#IMPORTING PACKAGES
import numpy as np
from numpy import ndarray
import math
import time
start_time = time.time()

#DEFINING FUNCTIONS

# 1) SIGMOID
def g(z):
    return 1/(1 + np.exp(-z))

# 2) HYPOTHESIS OF TRAINING EXAMPLE x
def NN_hypothesis(x, THETA):
    for l in range(0,L-1):
        theta_l = THETA[l]
        G[l+1] = np.dot(theta_l, A[l])
        A[l+1] = g(G[l+1])
    hyp_x = A[L-1]
    return hyp_x

# 3) COST FUNCTION FOR ERROR
def NN_cost_error(hyp,y):
    temp = hyp.shape[0]
    return np.product(y, np.log(hyp)) + np.product(1-y, np.log(1-hyp))

# 4) COST FUNCTION FOR REGULARIZATION
def NN_cost_regularisation(THETA, L):
    for l in range(0,L):


#LOADING FILES
# /home/rohan1297/Desktop/Link to SEMESTER VI/PRML/PRML_CIFAR/Python_Codes
X = np.matrix(np.load('/media/rohan1297/New Volume/Documents/Academic Material/ECE/SEMESTER VI/PRML/PRML_CIFAR/Files/Data_Reduced/x_train_reduce.npy'))
Y = np.matrix(np.load('/media/rohan1297/New Volume/Documents/Academic Material/ECE/SEMESTER VI/PRML/PRML_CIFAR/Files/Data/y_train.npy')).T

#DECLARING USEFUL VARIABLES
n = X.shape[0]
d = X.shape[1]
c = 10
L = 2
S = np.array([n,c])

#INITIALISATIONS
THETA = [{}]
for l in range(0,L-1):
    # temp = np.ndarray((S[l+1], S[l]+1), float64)
    temp = np.random.randn(S[l+1], S[l]+1)
    THETA[l] = temp

#FORWARD PROPAGATION
temp1 = 0.
temp2 = 0.
J = 0.0
for i in range(0,n):
    yi = Y[i,:]
    xi = X[i,:]
    hyp_xi = NN_hypothesis(xi, THETA)
    temp1 = NN_cost_error(yi, hyp_xi)
    temp2 = NN_cost_regularisation(THETA)
    J = J + temp1 + LAMBDA*temp2
J = J/m

#BACK PROPAGATION
for l in range(L-1,0);
    temp1 = np.dot(THETA[l].T, DELTA[l+1])
    temp2 =
    DELTA[l] = np.product(temp1, temp2)

print("--- %s seconds ---" % (time.time() - start_time))