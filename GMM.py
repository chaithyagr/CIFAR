import numpy as np
from numpy import ndarray
import math

import time
start_time = time.time()

def gaussian(u,C,x,d):
    C = np.diag(np.diag(C))
    C_inv = np.linalg.inv(C)
    C_det = np.linalg.det(C)
    t4 = np.matrix(x-u)
    t5 = np.matrix(C_inv)

    t7 = math.sqrt(pow(2*math.pi, d) * C_det)
    t8 = np.dot(np.dot(t4, t5), t4.T)
    t9 = (1.0/t7) * math.exp(-t8/2)
    return t9

x_train = np.matrix(np.load('/home/rohan1297/Documents/PRML_CIFAR/Files/X_train_reduce.npy'))
y_train = np.matrix(np.load('/home/rohan1297/Documents/PRML_CIFAR/Files/y_train.npy'))

#DECLARING USEFUL VARIABLES
n = x_train.shape[0]            #no. of training examples
d = x_train.shape[1]            #no. of features
c = 10                          #no. of classes
nc = n/c                        #no. of training examples in each class
ng = 3                         #no. gaussians in GMM for each class

weights =  ndarray((ng,c), np.float64)
means =  ndarray((d,ng,c), np.float64)
Cov_Matrices =  ndarray((d,d,ng,c), np.float64)

#SEPERATING DATA INTO CLASSES
x_sort = np.matrix(np.zeros(shape=(n,d)))
for i in range(0,c):
    temp1 = np.matrix(np.where(y_train == i))
    temp2 = temp1[1,:]
    temp3 = np.matrix(x_train[temp2, :])
    x_sort[i*nc:(i+1)*nc,:] = temp3

x_train = x_sort
for m in range(0,c):
    x = x_train[m*nc:(m+1)*nc,:]

    #INITIALISING GAUSSIANS
    u = np.random.randn(d,ng)
    Cmat = ndarray((d,d,ng), np.float64)
    for k in range(0,ng):
        temp1 = np.random.randn(d)
        temp2 = np.exp(temp1)
        Cmat[:,:,k] = np.diag(temp2)
    alpha = np.ones(shape=(ng))/ng

    #TRAINING THE GMM
    for iter in range(0,0):

        #FINDING POSTERIOR PROBABILITIES FOR EACH TRAINING EXAMPLE BELONGING TO EACH GAUSSIAN FUNCITON
        temp1 = np.zeros(shape=(nc,ng))
        for k in range(0, ng):
            uk = np.matrix(u[:, k])
            Ck = np.matrix(Cmat[:, :, k])
            for i in range(0,nc):
                xi = np.matrix(x[i,:])
                temp1[i,k] = gaussian(uk,Ck,xi,d)*alpha[k]
        temp2 = np.sum(temp1, axis=1)
        temp3 = np.divide(temp1.T,temp2.T)
        p_xi_in_k = np.matrix(temp3.T)

        #RECOMPUTING THE PARAMETERS FOR EACH GAUSSIAN
        for k in range(0, ng):
            temp_alphak = np.mean(p_xi_in_k[:,k])

            temp_uk = (np.dot(x.T, p_xi_in_k[:,k])) / (temp_alphak*nc)
            temp_uk = np.array(temp_uk)

            temp = np.zeros(shape=(d,d))
            uk = u[:,k]
            for i in range(0,nc):
                xi = np.matrix(x[i, :])
                temp = temp + (np.outer(xi-uk, xi-uk) * p_xi_in_k[i,k])
            temp_Ck = temp/ (temp_alphak*nc)

            alpha[k] = temp_alphak
            u[:,k] = temp_uk[:,0]
            Cmat[:,:,k] = temp_Ck

    weights[:,m] = alpha
    means[:,:,m] = u
    Cov_Matrices[:,:,:,m] = Cmat

#SAVING PARAMETERS INTO FILES
np.save('/home/rohan1297/Documents/PRML_CIFAR/Files/weights.npy',weights)
np.save('/home/rohan1297/Documents/PRML_CIFAR/Files/means.npy',means)
np.save('/home/rohan1297/Documents/PRML_CIFAR/Files/Cov_Matrices.npy',Cov_Matrices)

print("--- %s seconds ---" % (time.time() - start_time))