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

#RANDOM SUBSET OF TRAINING DATA FOR LDA
n = x_train.shape[0]
indices = np.random.randint(0,n,30000)
X = x_train[indices,:]
y = y_train[indices]

#PERFORMING LDA TO REDUCE FEATURE DIMENSION
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()
model = lda.fit(X,y)
x_train_reduce = model.transform(x_train)
x_test_reduce = model.transform(x_test)

#STORING THE REDUCED TRAINING AND TEST DATA
np.save('/home/rohan1297/Documents/PRML_CIFAR/Files/X_train_reduce.npy',x_train_reduce)
np.save('/home/rohan1297/Documents/PRML_CIFAR/Files/X_test_reduce.npy',x_test_reduce)
np.save('/home/rohan1297/Documents/PRML_CIFAR/Files/y_train.npy',y_train)
np.save('/home/rohan1297/Documents/PRML_CIFAR/Files/y_test.npy',y_test)