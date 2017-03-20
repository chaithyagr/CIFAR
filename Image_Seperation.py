import numpy as np
import math
import scipy.misc

import time
start_time = time.time()

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
atest = unpickle('/home/rohan1297/Documents/PRML_CIFAR/cifar-10-batches-py/test_batch')

#CONVERTING IMAGES INTO NUMPY MATRICES
x1 = np.matrix(np.array(a1['data']))
x2 = np.matrix(np.array(a2['data']))
x3 = np.matrix(np.array(a3['data']))
x4 = np.matrix(np.array(a4['data']))
x5 = np.matrix(np.array(a5['data']))
y1 = np.matrix(np.array(a1['labels']))
y2 = np.matrix(np.array(a2['labels']))
y3 = np.matrix(np.array(a3['labels']))
y4 = np.matrix(np.array(a4['labels']))
y5 = np.matrix(np.array(a5['labels']))

x_test = np.matrix(np.array(atest['data']))
y_test = np.matrix(np.array(atest['labels']))
x_train = np.concatenate((x1, x2,x3,x4,x5))
y_train = np.concatenate((y1.T,y2.T,y3.T,y4.T,y5.T))

#CUZ I DID SOME SHIT
y_train = np.array(y_train[:,0])
y_train = np.array(y_train[:,0])

n = x_train.shape[0]
for i in range(0,100):
    temp1 = x_train[i,:]
    temp2 = np.array(temp1)
    temp3 = np.reshape(temp2, (32,32,3), order='F')
    x = np.reshape(temp3, (32,32,3))
    scipy.misc.imsave('/home/rohan1297/Documents/PRML_CIFAR/Images/image'+ str(i) + '.jpg', x)

print("--- %s seconds ---" % (time.time() - start_time))