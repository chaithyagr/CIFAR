import time
start_time = time.time()

import numpy as np
import math
import scipy.io as sio

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()

##################  LDA ON PIXELS  ###############
x_train = np.load('/media/rohan1297/New Volume1/Documents/Academic Material/ECE/SEMESTER VI/PRML/PRML_CIFAR/Files/Data/x_train.npy')
y_train = np.load('/media/rohan1297/New Volume1/Documents/Academic Material/ECE/SEMESTER VI/PRML/PRML_CIFAR/Files/Data/y_train.npy')
# x_cv = np.load('/media/rohan1297/New Volume1/Documents/Academic Material/ECE/SEMESTER VI/PRML/PRML_CIFAR/Files/Data/x_cv.npy')
x_test = np.load('/media/rohan1297/New Volume1/Documents/Academic Material/ECE/SEMESTER VI/PRML/PRML_CIFAR/Files/Data/x_test.npy')
#RANDOM SUBSET OF TRAINING DATA FOR LDA
n = x_train.shape[0]
indices = np.random.randint(0,n,20000)
X = x_train[indices,:]
y = y_train[indices]
#PERFORMING LDA TO REDUCE FEATURE DIMENSION
model = lda.fit(X,y)
x_train_reduce_pixels = model.transform(x_train)
# x_cv_reduce_pixels = model.transform(x_cv)
x_test_reduce_pixels = model.transform(x_test)


##################  LDA ON HOG  ###############
temp1 = sio.loadmat('/media/rohan1297/New Volume1/Documents/Academic Material/ECE/SEMESTER VI/PRML/PRML_CIFAR/Files/Feature_Data/x_hog_train.mat')
# temp2 = sio.loadmat('/media/rohan1297/New Volume1/Documents/Academic Material/ECE/SEMESTER VI/PRML/PRML_CIFAR/Files/Feature_Data/x_hog_cv.mat')
temp3 = sio.loadmat('/media/rohan1297/New Volume1/Documents/Academic Material/ECE/SEMESTER VI/PRML/PRML_CIFAR/Files/Feature_Data/x_hog_test.mat')
x_train = temp1['x_hog_train']
# x_cv = temp2['x_hog_cv']
x_test = temp3['x_hog_test']
#PERFORMING LDA TO REDUCE FEATURE DIMENSION
X = x_train
y = y_train
model = lda.fit(X,y)
x_train_reduce_HOG = model.transform(x_train)
# x_cv_reduce_HOG = model.transform(x_cv)
x_test_reduce_HOG = model.transform(x_test)

# ##################  LDA ON LOG  ###############
# temp1 = sio.loadmat('media/rohan1297/New Volume1/Documents/Academic Material/ECE/SEMESTER VI/PRML/PRML_CIFAR/Files/Feature_Data/x_log_train.mat')
# temp2 = sio.loadmat('media/rohan1297/New Volume1/Documents/Academic Material/ECE/SEMESTER VI/PRML/PRML_CIFAR/Files/Feature_Data/x_log_cv.mat')
# temp2 = sio.loadmat('media/rohan1297/New Volume1/Documents/Academic Material/ECE/SEMESTER VI/PRML/PRML_CIFAR/Files/Feature_Data/x_log_test.mat')
# x_train = temp1['x_log_train']
# x_test = temp2['x_log_test']
# #PERFORMING LDA TO REDUCE FEATURE DIMENSION
# X = x_train
# y = y_train
# model = lda.fit(X,y)
# x_train_reduce_LOG = model.transform(x_train)
# x_test_reduce_LOG = model.transform(x_test)

#SAVING REDUCED FEATURE TRAINING AND TEST DATA
x_train = np.hstack((x_train_reduce_pixels, x_train_reduce_HOG))
# x_cv = np.hstack((x_cv_reduce_pixels, x_cv_reduce_HOG))
x_test = np.hstack((x_test_reduce_pixels, x_test_reduce_HOG))
np.save('/media/rohan1297/New Volume1/Documents/Academic Material/ECE/SEMESTER VI/PRML/PRML_CIFAR/Files/Data_Reduced/x_train_reduce.npy', x_train)
# np.save('/media/rohan1297/New Volume1/Documents/Academic Material/ECE/SEMESTER VI/PRML/PRML_CIFAR/Files/Data_Reduced/x_cv_reduce.npy', x_cv)
np.save('/media/rohan1297/New Volume1/Documents/Academic Material/ECE/SEMESTER VI/PRML/PRML_CIFAR/Files/Data_Reduced/x_test_reduce.npy', x_test)

print("--- %s seconds ---" % (time.time() - start_time))