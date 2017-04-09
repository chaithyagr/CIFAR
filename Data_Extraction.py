import numpy as np
import math
import scipy.misc
import scipy.io as sio
import sys
from sys import getsizeof

import time
start_time = time.time()

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict
#READING DATA FROM FILE
a1 = unpickle('/media/rohan1297/New Volume1/Documents/Academic Material/ECE/SEMESTER VI/PRML/PRML_CIFAR/cifar-10-batches-py/data_batch_1')
a2 = unpickle('/media/rohan1297/New Volume1/Documents/Academic Material/ECE/SEMESTER VI/PRML/PRML_CIFAR/cifar-10-batches-py/data_batch_2')
a3 = unpickle('/media/rohan1297/New Volume1/Documents/Academic Material/ECE/SEMESTER VI/PRML/PRML_CIFAR/cifar-10-batches-py/data_batch_3')
a4 = unpickle('/media/rohan1297/New Volume1/Documents/Academic Material/ECE/SEMESTER VI/PRML/PRML_CIFAR/cifar-10-batches-py/data_batch_4')
a5 = unpickle('/media/rohan1297/New Volume1/Documents/Academic Material/ECE/SEMESTER VI/PRML/PRML_CIFAR/cifar-10-batches-py/data_batch_5')
atest = unpickle('/media/rohan1297/New Volume1/Documents/Academic Material/ECE/SEMESTER VI/PRML/PRML_CIFAR/cifar-10-batches-py/test_batch')

#CONVERTING IMAGES INTO NUMPY MATRICES
x1 = np.array(a1['data'])
x2 = np.array(a2['data'])
x3 = np.array(a3['data'])
x4 = np.array(a4['data'])
x5 = np.array(a5['data'])
y1 = np.array(a1['labels'])
y2 = np.array(a2['labels'])
y3 = np.array(a3['labels'])
y4 = np.array(a4['labels'])
y5 = np.array(a5['labels'])

x_train = np.vstack((x1, x2,x3,x4,x5))
y_train = np.hstack((y1,y2,y3,y4,y5))
x_test = np.array(atest['data'])
y_test = np.array(atest['labels'])

# #DECLARING USEFUL VARIABLES
# c = 10                          #no. of classes
# n = x_train.shape[0]            #no. of training examples
# d = x_train.shape[1]                  #no. of features
#
# # #cvp = input("enter percentage of data for cross validation: ")
# # cv_percent = 20
#
# #FINDING NUMBER OF SAMPLES PER CLASS AND SEPERATING DATA INTO CLASSES
# x_sort = np.matrix(np.zeros(shape=(n, d)))
# nc = np.ndarray((c), np.int)
# j = 0
# for i in range(0,c):
#     temp1 = np.where(y == i)*1
#     nc[i] = np.sum(y == i)
#     temp2 = x[temp1, :]
#     x_sort[j:j+nc[i],:] = np.array(temp2[0,:,:])
#     j = j + nc[i]

# #CREATING TRAINING AND CROSS VALIDATION SETS
# n_cv_c = nc*cv_percent/100
# n_train_c = nc - n_cv_c
# ntrain = np.sum(n_train_c)
# ncv = np.sum(n_cv_c)
#
# j = 0
# ind_train = np.ndarray((0), int)
# ind_cv = np.ndarray((0), int)
# for i in range(0,c):
#     temp1 = np.arange(nc[i])
#     temp2 = j + np.random.permutation(temp1)
#     temp3 = temp2[0:n_train_c[i]]
#     temp4 = temp2[n_train_c[i]:nc[i]]
#     ind_train = np.hstack((ind_train, temp3))
#     ind_cv = np.hstack((ind_cv, temp4))
#     j = j+nc[i]
#
# x_train = x_sort[ind_train,:]
# x_cv = x_sort[ind_cv, :]
#
# y_train = np.ndarray((0), int)
# y_cv = np.ndarray((0), int)
# for i in range(0,c):
#     y_train = np.hstack((y_train, i* np.ones(n_train_c[i])))
#     y_cv = np.hstack((y_cv, i * np.ones(n_cv_c[i])))
#
# #SHUFFLING DATA
# indices_train = np.random.permutation(ntrain)
# indices_cv = np.random.permutation(ncv)
# x_train = x_train[indices_train, :]
# x_cv = x_cv[indices_cv, :]
# y_train = y_train[indices_train]
# y_cv = y_cv[indices_cv]
#
# #COMPUTING PRIOR PROBABILITIES FOR EACH CLASS
# priors = nc*1.0/n

#SAVING AS NUMPY FILES
np.save('/media/rohan1297/New Volume1/Documents/Academic Material/ECE/SEMESTER VI/PRML/PRML_CIFAR/Files/Data/x_train.npy',x_train)
np.save('/media/rohan1297/New Volume1/Documents/Academic Material/ECE/SEMESTER VI/PRML/PRML_CIFAR/Files/Data/y_train.npy',y_train)
# np.save('/media/rohan1297/New Volume1/Documents/Academic Material/ECE/SEMESTER VI/PRML/PRML_CIFAR/Files/Data/x_cv.npy',x_cv)
# np.save('/media/rohan1297/New Volume1/Documents/Academic Material/ECE/SEMESTER VI/PRML/PRML_CIFAR/Files/Data/y_cv.npy',y_cv)
np.save('/media/rohan1297/New Volume1/Documents/Academic Material/ECE/SEMESTER VI/PRML/PRML_CIFAR/Files/Data/x_test.npy',x_test)
np.save('/media/rohan1297/New Volume1/Documents/Academic Material/ECE/SEMESTER VI/PRML/PRML_CIFAR/Files/Data/y_test.npy',y_test)
# np.save('/media/rohan1297/New Volume1/Documents/Academic Material/ECE/SEMESTER VI/PRML/PRML_CIFAR/Files/Data/priors.npy',priors)

# #SAVING DATA IN MATLAB FILES
sio.savemat('/media/rohan1297/New Volume1/Documents/Academic Material/ECE/SEMESTER VI/PRML/PRML_CIFAR/Files/Data/x_train.mat', {'vect':x_train})
sio.savemat('/media/rohan1297/New Volume1/Documents/Academic Material/ECE/SEMESTER VI/PRML/PRML_CIFAR/Files/Data/y_train.mat', {'vect':y_train})
# sio.savemat('/media/rohan1297/New Volume1/Documents/Academic Material/ECE/SEMESTER VI/PRML/PRML_CIFAR/Files/Data/x_cv.mat', {'vect':x_cv})
# sio.savemat('/media/rohan1297/New Volume1/Documents/Academic Material/ECE/SEMESTER VI/PRML/PRML_CIFAR/Files/Data/y_cv.mat', {'vect':y_cv})
sio.savemat('/media/rohan1297/New Volume1/Documents/Academic Material/ECE/SEMESTER VI/PRML/PRML_CIFAR/Files/Data/x_test.mat', {'vect':x_test})
sio.savemat('/media/rohan1297/New Volume1/Documents/Academic Material/ECE/SEMESTER VI/PRML/PRML_CIFAR/Files/Data/y_test.mat', {'vect':y_test})

print("--- %s seconds ---" % (time.time() - start_time))