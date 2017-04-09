import numpy as np
from numpy import ndarray
import math
import matplotlib.pyplot as plt

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

# acc_n_iter = np.ndarray(20)
# for n_iter in range (0,20):

#READING TEST DATA AND PARAMETERS FROM FILE
x_test = np.matrix(np.load('/media/rohan1297/New Volume1/Documents/Academic Material/ECE/SEMESTER VI/PRML/PRML_CIFAR/Files/Data_Reduced/x_test_reduce.npy'))
y_test = (np.matrix(np.load('/media/rohan1297/New Volume1/Documents/Academic Material/ECE/SEMESTER VI/PRML/PRML_CIFAR/Files/Data/y_test.npy'))).T
weights = np.load('/media/rohan1297/New Volume1/Documents/Academic Material/ECE/SEMESTER VI/PRML/PRML_CIFAR/Files/Parameters/weights.npy')
means = np.load('/media/rohan1297/New Volume1/Documents/Academic Material/ECE/SEMESTER VI/PRML/PRML_CIFAR/Files/Parameters/means.npy')
Cov_Matrices = np.load('/media/rohan1297/New Volume1/Documents/Academic Material/ECE/SEMESTER VI/PRML/PRML_CIFAR/Files/Parameters/Cov_Matrices.npy')

#DECLARING USEFUL VARIABLES
nt = x_test.shape[0]
c = weights.shape[1]
ng = weights.shape[0]
d = means.shape[0]
ntc = nt/c

#CLASSIFYING THE TEST DATA
temp1 =  ndarray((nt,ng,c), np.float64)
for i in range(0,nt):
    xi = x_test[i,:]
    for k in range(0,c):
        for j in range(0,ng):
            u = means[:,j,k]
            C = Cov_Matrices[:,:,j,k]
            temp1[i,j,k] = gaussian(u,C,xi,d) * weights[j,k]

temp2 = np.sum(temp1,axis=1)
y_predict = np.matrix(np.argmax(temp2,axis=1)).T

correct_classifications = np.sum(y_predict == y_test)
misclassifications = nt - correct_classifications
accuracy = correct_classifications*100.0/nt

class_accuracy = np.ndarray((c), np.float64)
for i in range(0,c):
    temp1 = (y_predict == i)*1
    temp2 = (y_test == i)*1
    temp3 = np.dot(temp1.T,temp2)
    class_accuracy[i] = (temp3*100.0)/ntc
    print 'Accuracy for class %d =%f' % i, accuracy[i]

    # class_accuracies = ndarray(c, np.float64)
    # for i in range(0,c):
    #     temp1 = (y_predict == i)*1
    #     temp2 = (y_test == i)*1
    #     temp3 = np.dot(temp1.T,temp2)
    #     class_accuracies[i] = temp3*100.0/nt
    #     print 'Accuracy for Class %d = %f' % (i, class_accuracies[i])
    #
    # acc_n_iter[n_iter] = accuracy

# n_interations = np.arange(0,20)
# plt.plot(n_interations, acc_n_iter)
# plt.xlabel('No. of Iterations for training the GMM')
# plt.ylabel('Prediction Accuracy in Percentage')
# plt.show()


print 'Correct Classifications = %d' % correct_classifications
print 'Wrong Classifications = %d' % misclassifications
print 'Accuracy in percentage = %f' % accuracy

print("--- %s seconds ---" % (time.time() - start_time))