import numpy as np
from numpy import ndarray
import math
import matplotlib.pyplot as plt

import time
start_time = time.time()

# 1) SIGMOID
def g(z):
    return 1/(1 + np.exp(-z))

X_temp = np.load('/media/rohan1297/New Volume/Documents/Academic Material/ECE/SEMESTER VI/PRML/PRML_CIFAR/Files/Data_Reduced/x_test_reduce.npy')
Y_temp = np.load('/media/rohan1297/New Volume/Documents/Academic Material/ECE/SEMESTER VI/PRML/PRML_CIFAR/Files/Data/y_test.npy').T
THETA = np.load('/media/rohan1297/New Volume/Documents/Academic Material/ECE/SEMESTER VI/PRML/PRML_CIFAR/Files/Parameters/parameters.npy')

#DECLARING USEFUL VARIABLES
nt = X_temp.shape[0]
d = X_temp.shape[1]
c = 10
L = 2
S = np.array([d,c])

#APPENDING BIAS FEATURE TO TRAINING EXAMPLES
temp = np.ones((nt,1))
X = np.hstack((temp,X_temp))
#MAKING Y AS A VECTOR FOR EACH TRAINING EXAMPLE
Y = np.zeros((nt,c))
for i in range(0,nt):
    Y[i,Y_temp[i]] = 1

#INITIALISING SOME VARIABLES
A = [{}] * L
Z = [{}] * L
for l in range(0,L):
    temp = np.array(np.random.randn(S[l]))
    Z[l] = temp
for l in range(0,L):
    temp1 = np.random.randn(S[l])
    temp2 = np.array([1])
    A[l] = np.hstack((temp1, temp2))


#CLASSIFYING EACH TEST SAMPLE
Y_predict = np.zeros((nt,c))

for i in range(0,nt):
    y = Y[i, :]
    x = X[i, :]

    A[0] = x
    for l in range(1,L):
        Z[l] = np.dot(THETA[l-1], A[l-1])
        A[l][1:] = g(Z[l])

    temp = np.argmax(A[L-1])
    Y_predict[i,temp] = 1

print("--- %s seconds ---" % (time.time() - start_time))