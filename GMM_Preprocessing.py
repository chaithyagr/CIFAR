import numpy as np
import math

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

#READING DATA FROM FILE
a1 = unpickle('/home/rohan1297/Documents/PRML_CIFAR/cifar-10-batches-py/data_batch_1')
a2 = unpickle('/home/rohan1297/Documents/PRML_CIFAR/cifar-10-batches-py/data_batch_2')
a3 = unpickle('/home/rohan1297/Documents/PRML_CIFAR/cifar-10-batches-py/data_batch_3')
a4 = unpickle('/home/rohan1297/Documents/PRML_CIFAR/cifar-10-batches-py/data_batch_4')
a5 = unpickle('/home/rohan1297/Documents/PRML_CIFAR/cifar-10-batches-py/data_batch_5')

#CONVERTING IMAGES INTO NUMPY MATRICES
x1 = np.asmatrix(np.array(a1['data']))
x2 = np.asmatrix(np.array(a2['data']))
x3 = np.asmatrix(np.array(a3['data']))
x4 = np.asmatrix(np.array(a4['data']))
x5 = np.asmatrix(np.array(a5['data']))
y1 = np.asmatrix(np.array(a1['labels']))
y2 = np.asmatrix(np.array(a2['labels']))
y3 = np.asmatrix(np.array(a3['labels']))
y4 = np.asmatrix(np.array(a4['labels']))
y5 = np.asmatrix(np.array(a5['labels']))
x_train = np.concatenate((x1, x2,x3,x4,x5))
y = np.concatenate((y1.T,y2.T,y3.T,y4.T,y5.T))

#DECLARING USEFUL VARIABLES
n = x_train.shape[0]          #no. of training examples
d = x_train.shape[1]          #no. of features
c = 10                  #no. of classes
nc = n/c                #no. of training examples in each class
ng = 10                  #no. gaussians in GMM for each class

#SEPERATING DATA INTO CLASSES
x_sort = np.asmatrix(np.zeros(shape=(n,d)))
for i in range(0,c):
    temp1 = np.asmatrix(np.where(y == i))
    temp2 = temp1[0,:]
    temp3 = np.asmatrix(x_train[temp2, :])
    temp4 = temp3[0, :, :]
    x_sort[i*nc:(i+1)*nc,:] = temp4

#REMOVING REDUNDANCY IN DATA USING PCA
for i in range(0,10):
    #PRE PROCESSING DATA
    x = x_sort[0:nc,:]
    u = np.mean(x, axis=0)                      #mean vector of features
    sigma = np.std(x, axis=0)                   #standard deviation of features
    temp = np.subtract(x,u)                     #mean normalisation
    x = np.divide(temp, sigma)                 #feature scaling

    #DOING PCA TO REDUCE NO. OF FEATURES
    C = np.dot(x.T, x)/nc                       #covariance matrix of data
    U, S, V = np.linalg.svd(C)                  #doing svd of covariance matrix
    threshold  = 0.80*np.sum(S)                 #threshold for amount of variance to be retained
    k = 1                                       #no. of features required
    while np.sum(S[0:k]) <= threshold :
           k = k+1
    Ureduce = U[:,1:k]                                 #getting required eigenvectors
    x_reduced = np.dot(x,Ureduce)    #getting dataset with reduced features
    np.savetxt("data_reduce"+str(i)+".csv", x_reduced, delimiter=",")         #storing reduced feature dataset in a file