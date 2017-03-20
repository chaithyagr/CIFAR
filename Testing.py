import numpy as np
from numpy import ndarray
import math

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

#READING TEST DATA AND PARAMETERS FROM FILE
x_test = np.matrix(np.load('X_test_reduce.npy'))
y_test = (np.matrix(np.load('y_test.npy'))).T
weights = np.load('weights.npy')
means = np.load('means.npy')
Cov_Matrices = np.load('Cov_Matrices.npy')

#DECLARING USEFUL VARIABLES
nt = x_test.shape[0]
c = weights.shape[1]
ng = weights.shape[0]
d = means.shape[0]

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

print 'Correct Classifications = %d' % correct_classifications
print 'Wrong Classifications = %d' % misclassifications
print 'Correct_Classifications = %d' % accuracy