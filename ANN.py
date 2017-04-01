#IMPORTING PACKAGES
import numpy as np
from numpy import ndarray
import math
import time
start_time = time.time()

# #DEFINING FUNCTIONS

# 1) SIGMOID
def g(z):
    return 1/(1 + np.exp(-z))

# # 2) HYPOTHESIS OF TRAINING EXAMPLE x
# def NN_hypothesis(x, THETA):
#     for l in range(0,L-1):
#         theta_l = THETA[l]
#         G[l+1] = np.dot(theta_l, A[l])
#         A[l+1] = g(G[l+1])
#     hyp_x = A[L-1]
#     return hyp_x
#
# # 3) COST FUNCTION FOR ERROR
# def NN_cost_error(hyp,y):
#     temp = hyp.shape[0]
#     return np.product(y, np.log(hyp)) + np.product(1-y, np.log(1-hyp))
#
# # 4) COST FUNCTION FOR REGULARIZATION
# # def NN_cost_regularisation(THETA, L):
# #     for l in range(0,L):


# LOADING FILES
X_temp = np.load('/media/rohan1297/New Volume/Documents/Academic Material/ECE/SEMESTER VI/PRML/PRML_CIFAR/Files/Data_Reduced/x_train_reduce.npy')
Y_temp = np.load('/media/rohan1297/New Volume/Documents/Academic Material/ECE/SEMESTER VI/PRML/PRML_CIFAR/Files/Data/y_train.npy').T

#DECLARING USEFUL VARIABLES
n = X_temp.shape[0]
d = X_temp.shape[1]
c = 10
L = 2
S = np.array([d,c])
alpha = 0.01                #learning rate

#APPENDING BIAS FEATURE TO TRAINING EXAMPLES
temp = np.ones((n,1))
X = np.hstack((temp,X_temp))
#MAKING Y AS A VECTOR FOR EACH TRAINING EXAMPLE
Y = np.zeros((n,c))
for i in range(0,n):
    Y[i,Y_temp[i]] = 1

#INITIALISATIONS
THETA = [{}] * (L-1)
DELTA = [{}] * (L-1)
delta = [{}] * L
A = [{}] * L
Z = [{}] * L

for l in range(0,L-1):
    temp = np.random.randn(S[l+1], S[l]+1) * 0.0001
    THETA[l] = temp
for l in range(0,L):
    temp = np.array(np.random.randn(S[l]))
    Z[l] = temp
for l in range(0,L):
    temp1 = np.random.randn(S[l])
    temp2 = np.array([1])
    A[l] = np.hstack((temp1, temp2))

##################......TRAINING THE NEURAL NETWORK...............###############

for iter in range(0,10):

    for l in range(0, L-1):
        DELTA[l] = THETA[l] * 0.0

    for i in range(0,n):
        y = Y[i, :]
        x = X[i, :]

        A[0] = x

        for l in range(1,L):
            Z[l] = np.dot(THETA[l-1], A[l-1])
            A[l][1:] = g(Z[l])

        delta[L-1] = A[L-1][1:] - y
        for l in range(L-2, -1, -1):
            temp1 = np.dot(THETA[l][:, 1:].T, delta[l+1])
            temp2 = A[l][1:] * (1-A[l][1:])
            delta[l] = temp1 * temp2

        for l in range(0, L-1):
            DELTA[l] = DELTA[l] + np.outer(delta[l+1], A[l])

    for l in range(0,L-1):
        THETA[l] = THETA[l] - ((alpha/n) * DELTA[l])

    # #FORWARD PROPAGATION
    # temp1 = 0.
    # temp2 = 0.
    # J = 0.0
    # hyp_xi = NN_hypothesis(xi, THETA)
    # temp1 = NN_cost_error(yi, hyp_xi)
    # temp2 = NN_cost_regularisation(THETA)
    # J = J + temp1 + LAMBDA*temp2
    # J = J/m

np.save('/media/rohan1297/New Volume/Documents/Academic Material/ECE/SEMESTER VI/PRML/PRML_CIFAR/Files/Parameters/parameters.npy',THETA)



print("--- %s seconds ---" % (time.time() - start_time))